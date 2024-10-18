from pathlib import Path

import pandas as pd
import numpy as np
from model.analysis import Analysis
from model.state import CancerState, HpvState, HpvStrain, LifeState

from src.helper_functions import combine_age_groups
import logging

def run_analysis(scenario_dir: Path, iteration: int):
    """
    Run analysis for the USA scenario and generate target values.

    This function performs various analyses on HPV prevalence, CIN2/3 prevalence,
    cancer incidence, and cause of cancer. It generates target values for different
    age groups and HPV strains, and saves the results to a CSV file.

    Args:
        scenario_dir (Path): The directory path for the scenario being analyzed.
        iteration (int): The iteration number of the analysis.

    Returns:
        None

    Side effects:
        - Creates a CSV file named 'analysis_values.csv' in the iteration directory.
        - Logs information and debug messages using the logging module.

    The function performs the following analyses:
    1. HPV Prevalence for low-risk, high-risk, HPV-16, and HPV-18 strains.
    2. CIN2/3 Prevalence for high-risk, HPV-16, and HPV-18 strains.
    3. Cancer Incidence.
    4. Cause of Cancer for HPV-16, HPV-18, and high-risk strains.

    Results are combined into a single DataFrame and saved as a CSV file.
    """
    
    logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for more detailed logs
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting USA analysis for scenario: {scenario_dir}, iteration: {iteration}")
    
    analysis = Analysis(scenario_dir, iteration)

    hpv_age_groups = [20, 30, 40, 50, 60, 100]
    cin23_age_groups = [20, 25, 30, 40, 50, 60, 80, 100]
    cancer_age_groups = [20, 30, 40, 50, 60, 70, 100]

    results = []

    # HPV Prevalence Targets
    for strain, target in [
        (HpvStrain.LOW_RISK, "HPV_LR"),
        (HpvStrain.HIGH_RISK, "HPV_HR"),
        (HpvStrain.SIXTEEN, "HPV_16"),
        (HpvStrain.EIGHTEEN, "HPV_18")
    ]:
        df = analysis.prevalence(field=strain.name, states=(HpvState.HPV.value,))
        results.append(combine_age_groups(df=df * 100, ages=hpv_age_groups, target=target).iloc[:4])

    # CIN2 and CIN3 Prevalence Targets
    for strain, target_prefix in [
        (HpvStrain.HIGH_RISK, "CIN"),
        (HpvStrain.SIXTEEN, "CIN"),
        (HpvStrain.EIGHTEEN, "CIN")
    ]:
        for cin_state, cin_suffix in [(HpvState.CIN_2, "2"), (HpvState.CIN_3, "3")]:
            df = analysis.prevalence(field=strain.name, states=(cin_state.value,))
            target = f"{target_prefix}{cin_suffix}_{strain.name.split('_')[-1]}"
            results.append(combine_age_groups(df=df * 100, ages=cin23_age_groups, target=target).iloc[:6])

    # Cancer Incidence Targets
    df = 500_000 * analysis.incidence(CancerState.id, CancerState.LOCAL.value)
    results.append(combine_age_groups(df=df, ages=cancer_age_groups, target="Cancer_Inc"))

    # Cause of Cancer
    cancer_counts = {
        strain: np.sum(analysis.agent_events[strain.name]["To"] == HpvState.CANCER.value)
        for strain in [HpvStrain.SIXTEEN, HpvStrain.EIGHTEEN, HpvStrain.HIGH_RISK]
    }
    cancer_total = max(sum(cancer_counts.values()), 1)
    for strain, count in cancer_counts.items():
        results.append(pd.DataFrame({
            "Target": [f"Cause of Cancer: {strain.name}"],
            "Age": ["N/A"],
            "Model": [count / cancer_total]
        }))

    # Combine all results
    results_df = pd.concat(results, ignore_index=True)

    # Save as CSV
    output_file = analysis.iteration_dir.joinpath("analysis_values.csv")
    logger.info(f"Saving analysis_values.csv to {output_file}")
    results_df.columns = ["Target", "Age", str(iteration)]
    results_df.to_csv(output_file, index=False)
    logger.info("Analysis complete")
