import pytest
from unittest.mock import Mock
from model.screening import (
    ScreeningStrategy,
    ScreeningState,
    ScreeningTestResult,
    CervicalLesionState,
    HpvStrain,
)


@pytest.fixture
def strategy():
    model = Mock()
    params = Mock()
    return ScreeningStrategy(model, params)


def test_triage_cytology_negative(strategy):
    strategy.model.cervical_lesion.values = {1: CervicalLesionState.NORMAL}
    result = strategy.triage_cytology_results(1, ScreeningTestResult.NEGATIVE)
    assert result == ScreeningState.ROUTINE


def test_triage_cytology_ascus(strategy):
    strategy.model.cervical_lesion.values = {1: CervicalLesionState.ASCUS}
    result = strategy.triage_cytology_results(1, ScreeningTestResult.POSITIVE)
    assert result == ScreeningState.RE_TEST


def test_triage_cytology_lsil(strategy):
    strategy.model.cervical_lesion.values = {1: CervicalLesionState.LSIL}
    result = strategy.triage_cytology_results(1, ScreeningTestResult.POSITIVE)
    assert result == ScreeningState.COLPOSCOPY


def test_triage_cytology_hsil(strategy):
    strategy.model.cervical_lesion.values = {1: CervicalLesionState.HSIL}
    result = strategy.triage_cytology_results(1, ScreeningTestResult.POSITIVE)
    assert result == ScreeningState.COLPOSCOPY


def test_triage_dna_all_negative(strategy):
    result = strategy.triage_dna_results(
        1, {strain: ScreeningTestResult.NEGATIVE for strain in HpvStrain}
    )
    assert result == ScreeningState.ROUTINE


def test_triage_dna_hpv16_positive(strategy):
    result = strategy.triage_dna_results(
        1,
        {
            HpvStrain.SIXTEEN: ScreeningTestResult.POSITIVE,
            HpvStrain.EIGHTEEN: ScreeningTestResult.NEGATIVE,
            HpvStrain.HIGH_RISK: ScreeningTestResult.NEGATIVE,
            HpvStrain.LOW_RISK: ScreeningTestResult.NEGATIVE,
        },
    )
    assert result == ScreeningState.COLPOSCOPY


def test_triage_dna_hpv18_positive(strategy):
    result = strategy.triage_dna_results(
        1,
        {
            HpvStrain.SIXTEEN: ScreeningTestResult.NEGATIVE,
            HpvStrain.EIGHTEEN: ScreeningTestResult.POSITIVE,
            HpvStrain.HIGH_RISK: ScreeningTestResult.NEGATIVE,
            HpvStrain.LOW_RISK: ScreeningTestResult.NEGATIVE,
        },
    )
    assert result == ScreeningState.COLPOSCOPY


def test_triage_dna_other_hr_hpv_positive(strategy):
    result = strategy.triage_dna_results(
        1,
        {
            HpvStrain.SIXTEEN: ScreeningTestResult.NEGATIVE,
            HpvStrain.EIGHTEEN: ScreeningTestResult.NEGATIVE,
            HpvStrain.HIGH_RISK: ScreeningTestResult.POSITIVE,
            HpvStrain.LOW_RISK: ScreeningTestResult.NEGATIVE,
        },
    )
    assert result == ScreeningState.RE_TEST


def test_triage_cotest_all_negative(strategy):
    results = {
        "cotest": ScreeningTestResult.NEGATIVE,
        "cytology": ScreeningTestResult.NEGATIVE,
        "hr_hpv": False,
        "hpv_16_18": False,
    }
    result = strategy.triage_cotest_results(1, results)
    assert result == ScreeningState.ROUTINE


def test_triage_cotest_hr_hpv_positive_cytology_negative(strategy):
    results = {
        "cotest": ScreeningTestResult.POSITIVE,
        "cytology": ScreeningTestResult.NEGATIVE,
        "hr_hpv": True,
        "hpv_16_18": False,
    }
    result = strategy.triage_cotest_results(1, results)
    assert result == ScreeningState.RE_TEST


def test_triage_cotest_hr_hpv_positive_cytology_negative_hpv16_18_positive(strategy):
    results = {
        "cotest": ScreeningTestResult.POSITIVE,
        "cytology": ScreeningTestResult.NEGATIVE,
        "hr_hpv": True,
        "hpv_16_18": True,
    }
    result = strategy.triage_cotest_results(1, results)
    assert result == ScreeningState.COLPOSCOPY


def test_triage_cotest_hr_hpv_positive_cytology_positive(strategy):
    results = {
        "cotest": ScreeningTestResult.POSITIVE,
        "cytology": ScreeningTestResult.POSITIVE,
        "hr_hpv": True,
        "hpv_16_18": False,
    }
    result = strategy.triage_cotest_results(1, results)
    assert result == ScreeningState.COLPOSCOPY


def test_triage_cotest_hr_hpv_negative_cytology_positive_ascus(strategy):
    strategy.model.cervical_lesion.values = {1: CervicalLesionState.ASCUS}
    results = {
        "cotest": ScreeningTestResult.POSITIVE,
        "cytology": ScreeningTestResult.POSITIVE,
        "hr_hpv": False,
        "hpv_16_18": False,
    }
    result = strategy.triage_cotest_results(1, results)
    assert result == ScreeningState.RE_TEST


def test_triage_cotest_hr_hpv_negative_cytology_positive_hsil(strategy):
    strategy.model.cervical_lesion.values = {1: CervicalLesionState.HSIL}
    results = {
        "cotest": ScreeningTestResult.POSITIVE,
        "cytology": ScreeningTestResult.POSITIVE,
        "hr_hpv": False,
        "hpv_16_18": False,
    }
    result = strategy.triage_cotest_results(1, results)
    assert result == ScreeningState.COLPOSCOPY
