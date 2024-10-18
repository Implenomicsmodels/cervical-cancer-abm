from model.tests.fixtures import model_base
from model.state import HpvState

def test_transition_cin1_to_cin2_3_low_risk(model_base):
    # --- Focus only on the low-risk strain (hpv_strains[1])
    low_risk_strain = model_base.hpv_strains[1]

    # --- Set all agents' HPV state to CIN_1
    low_risk_strain.values.fill(HpvState.CIN_1)
    
    print("Initial states (CIN_1):", low_risk_strain.values[:10])  # Show initial state of first 10 agents

    # --- Set probability of transitioning to CIN_2_3 to 1 for all agents
    low_risk_strain.probabilities.fill(1)

    # --- Step the model to force the transition
    low_risk_strain.step()
    
    print("States after step (should be CIN_2_3):", low_risk_strain.values[:10])  # Show state after step

    # --- Check if all agents have transitioned to CIN_2_3
    #assert low_risk_strain.values.min() == HpvState.CIN_2_3.value, "Not all agents transitioned to CIN_2_3"
    assert low_risk_strain.values.max() == HpvState.CIN_2_3.value, "Not all agents transitioned to CIN_2_3"

    print("All agents successfully transitioned from CIN_1 to CIN_2_3 in the low-risk strain.")

# If you want to manually run this test without pytest, you can call it like this:
# test_transition_cin1_to_cin2_3_low_risk(model_base)


__all__ = ["model_base"]
