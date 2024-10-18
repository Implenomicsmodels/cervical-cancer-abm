import argparse
import csv
from itertools import chain
import pandas as pd
import numpy as np
from pathlib import Path


def main(batch: str, country: str):
    batch_dir = Path(f"experiments/{country}").joinpath(batch)
    # Collect all the individual results files into a single data structure.
    results = []
    for scenario_dir in batch_dir.glob("scenario_*"):
        for results_file in scenario_dir.glob("**/results.csv"):
            with results_file.open() as f:
                reader = csv.DictReader(f)
                result = next(reader)
                result["scenario"] = scenario_dir.name.replace("scenario_", "")
                results.append(result)

    # Compute descriptive stats by scenario.
    results_df = pd.DataFrame(results).set_index("scenario").apply(pd.to_numeric, errors="coerce")
    groups = results_df.groupby(level=0)

    means = groups.mean()
    stds = groups.std().add_prefix("std_")
    mins = groups.min().add_prefix("min_")
    maxes = groups.max().add_prefix("max_")

    # Combine all the stats, grouping them by variable.
    combined = pd.concat([means, stds, mins, maxes], axis=1)
    column_order = list(chain(*zip(means.columns, stds.columns, mins.columns, maxes.columns)))
    combined = combined[column_order]

    # Export wide format
    combined.to_csv(batch_dir.joinpath("combined_results_wide.csv"))

    # Create long format
    combined_long = combined.reset_index().melt(id_vars=['scenario'], var_name='metric', value_name='value')
    
    # Split the metric column into statistic and variable
    combined_long['statistic'] = combined_long['metric'].apply(lambda x: x.split('_')[0] if x.startswith(('std', 'min', 'max')) else 'mean')
    combined_long['variable'] = combined_long['metric'].apply(lambda x: '_'.join(x.split('_')[1:]) if x.startswith(('std', 'min', 'max')) else x)
    
    # Pivot to create separate columns for each statistic
    combined_long_pivot = combined_long.pivot_table(
        values='value', 
        index=['scenario', 'variable'], 
        columns='statistic', 
        aggfunc='first'
    ).reset_index()

    # Reorder columns, including only those that exist
    column_order = ['scenario', 'variable']
    for col in ['mean', 'std', 'min', 'max']:
        if col in combined_long_pivot.columns:
            column_order.append(col)
    combined_long_pivot = combined_long_pivot[column_order]

    # Export long format
    combined_long_pivot.to_csv(batch_dir.joinpath("combined_results_long.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create all input files for a batch of scenarios")
    parser.add_argument("batch", type=str, help="name of the batch directory")
    parser.add_argument("--country", type=str, default="all", help="The directory containing the experiment")
    args = parser.parse_args()

    if args.country == "all":
        run_list = []
        for country in ["zambia", "japan", "usa", "india"]:
            main(batch=args.batch, country=country)
    else:
        main(batch=args.batch, country=args.country)

    # if args.country != "zambia":
    #     d = Path('experiments/usa/transition_dictionaries/')
    #     pickle_files = sorted(d.glob('*p[1-3]*.pickle'))
    #     csv_files = [
    #         'cervical_cancer_incidence.csv',
    #         'hpv_cin_prevalence.csv',
    #         'mortality.csv'
    #     ]
    #     o = Path('experiments/usa/batch_10/')
    #     o.mkdir(parents=True, exist_ok=True)
    #     g = lambda x: x + np.random.normal(0.1 * x, 0.05 * x)
    #     for pickle_file, csv_file in zip(pickle_files, csv_files):
    #         df = pd.read_pickle(pickle_file)
    #         df['Run Value'] = df['Run Value'].apply(g)
    #         df.to_csv(o / csv_file, index=False)