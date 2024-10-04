from enum import IntEnum, unique
from typing import Dict

from model.event import Event
from model.parameters import ScreeningParameters
from model.state import CancerState
from model.state import CancerDetectionState
from model.state import HpvState
from model.state import HpvStrain
from model.state import CervicalLesionState


@unique
class ScreeningTestResult(IntEnum):
    NEGATIVE = 0
    POSITIVE = 1
    CANCER = 2


@unique
class ScreeningState(IntEnum):
    ROUTINE = 1
    RE_TEST = 2
    SURVEILLANCE = 3


@unique
class ScreeningTest(IntEnum):
    CYTOLOGY = 1
    HPV = 2
    COTEST = 3


class ViaScreeningTest:
    def __init__(self, model):
        self.model = model
        self.params = model.params.screening.via

    def get_result(
        self, true_hpv_state: HpvState, true_cancer_state: CancerState
    ) -> ScreeningTestResult:
        """Return the screening test result given a woman's most advanced HPV state and her cancer state.

        Properties of the test:
        - If the HpvState is NORMAL, HPV, or CIN_1 and the test is specific, then return NEGATIVE.
        - If the HpvState is NORMAL, HPV, or CIN_1 and the test isn't specific, then return POSITIVE.
        - If the HpvState is CIN_2_3 and the test is sensitive, then return POSITIVE.
        - If the HpvState is CIN_2_3 and the test isn't sensitive, then return NEGATIVE.
        - If the HpvState is CANCER, the CancerState is LOCAL, and the test is sensitive, then return CANCER.
        - If the HpvState is CANCER, the CancerState is LOCAL, and the test isn't sensitive, then return NEGATIVE.
        - If the HpvState is CANCER and the CancerState is REGIONAL or DISTANT, then return CANCER
            (regardless of the sensitivity).

        Raise a ValueError if true_cancer_state is DEAD. Also raise a ValueError
        if true_hpv_state is CANCER and true_cancer_state is NORMAL.
        """

        if true_cancer_state == CancerState.DEAD:
            raise ValueError()
        elif (
            true_hpv_state == HpvState.CANCER
            and true_cancer_state == CancerState.NORMAL
        ):
            raise ValueError()
        elif true_hpv_state in [HpvState.NORMAL, HpvState.HPV, HpvState.CIN_1]:
            if self.model.rng.random() > self.params.specificity:
                return ScreeningTestResult.POSITIVE
            else:
                return ScreeningTestResult.NEGATIVE
        elif true_hpv_state == HpvState.CIN_2_3:
            if self.model.rng.random() < self.params.sensitivity:
                return ScreeningTestResult.POSITIVE
            else:
                return ScreeningTestResult.NEGATIVE
        elif true_cancer_state == CancerState.LOCAL:
            if self.model.rng.random() < self.params.sensitivity:
                return ScreeningTestResult.CANCER
            else:
                return ScreeningTestResult.NEGATIVE
        else:
            return ScreeningTestResult.CANCER


class DnaScreeningTest:
    def __init__(self, model):
        self.model = model
        self.params = model.params.screening.dna

    def get_result(
        self, true_hpv_states: Dict[HpvStrain, HpvState]
    ) -> Dict[HpvStrain, ScreeningTestResult]:
        """Return the screening test result given a woman's true HPV state for each
            strain. An independent result is provided for each strain.

        Properties of the test:
        - Detectable HPV strains are SIXTEEN, EIGHTEEN, and HIGH_RISK.
        - We say that a woman "has a strain" when her state for that strain is
          one of HPV, CIN_1, CIN_2_3, or CANCER.
        - Always return NEGATIVE for undetectable strains.
        - If the woman has a detectable strain of HPV and the test is sensitive, then
          return POSITIVE for each detectable strain that she has.
        - If the woman has a detectable strain of HPV and the test isn't sensitive,
          then return NEGATIVE for all strains.
        - If the woman doesn't have a detectable strain of HPV and the test is
          specific, then return NEGATIVE for all strains.
        - If the woman doesn't have a detectable strain of HPV and the test isn't
          specific, then return POSITIVE for the HIGH_RISK strain and NEGATIVE for
          other strains.
        """

        detectable = [
            HpvStrain.SIXTEEN,
            HpvStrain.EIGHTEEN,
            HpvStrain.HIGH_RISK,
        ]

        # Step 1: Compute an overall positive/negative result using the test sensitivity and specificity.

        false_positive = False

        if all(true_hpv_states[strain] == HpvState.NORMAL for strain in detectable):
            if self.model.rng.random() > self.params.specificity:
                overall_result = ScreeningTestResult.POSITIVE
                false_positive = True
            else:
                overall_result = ScreeningTestResult.NEGATIVE
        else:
            if self.model.rng.random() < self.params.sensitivity:
                overall_result = ScreeningTestResult.POSITIVE
            else:
                overall_result = ScreeningTestResult.NEGATIVE

        if overall_result == ScreeningTestResult.NEGATIVE:
            return {strain: ScreeningTestResult.NEGATIVE for strain in HpvStrain}

        # Step 2: Compute strain-specific results using deterministic rules.

        result = {strain: ScreeningTestResult.NEGATIVE for strain in HpvStrain}

        for strain in detectable:
            if true_hpv_states[strain] != HpvState.NORMAL:
                result[strain] = ScreeningTestResult.POSITIVE

        if false_positive:
            result[HpvStrain.HIGH_RISK] = ScreeningTestResult.POSITIVE

        return result


class CytologyScreeningTest:
    def __init__(self, model):
        self.model = model
        self.params = model.params.screening.cytology

    def get_result(
        self, true_cervical_lesion_state: CervicalLesionState
    ) -> ScreeningTestResult:
        """Return the screening test result given a woman's true cervical lesion state.

        Properties of the test:
        - If the CervicalLesionState is NORMAL and the test is specific, then return NEGATIVE.
        - If the CervicalLesionState is NORMAL and the test isn't specific, then return POSITIVE.
        - If the CervicalLesionState is ASCUS, LSIL, ASCH, or HSIL and the test is sensitive, then return POSITIVE.
        - If the CervicalLesionState is ASCUS, LSIL, ASCH, or HSIL and the test isn't sensitive, then return NEGATIVE.
        """

        if true_cervical_lesion_state == CervicalLesionState.NORMAL:
            if self.model.rng.random() > self.params.specificity:
                return ScreeningTestResult.POSITIVE
            else:
                return ScreeningTestResult.NEGATIVE
        elif true_cervical_lesion_state in [
            CervicalLesionState.ASCUS,
            CervicalLesionState.LSIL,
            CervicalLesionState.ASCH,
            CervicalLesionState.HSIL,
        ]:
            if self.model.rng.random() < self.params.sensitivity:
                return ScreeningTestResult.POSITIVE
            else:
                return ScreeningTestResult.NEGATIVE
        else:
            raise ValueError(
                f"Unexpected cervical lesion state: {true_cervical_lesion_state}"
            )


class CoTestScreeningTest:
    def __init__(self, model):
        self.model = model
        self.params = model.params.screening.cotest
        self.cytology_test = CytologyScreeningTest(model)
        self.hpv_test = DnaScreeningTest(model)

    def get_result(self, unique_id: int) -> Dict[str, ScreeningTestResult]:
        age = self.model.age[unique_id]
        if age < 30 or age > 65:
            raise ValueError(
                f"CoTest is only applicable for ages 30-65. Current age: {age}"
            )

        cytology_result = self.cytology_test.get_result(unique_id)
        hpv_result = self.hpv_test.get_result(unique_id)

        # Determine the true disease state
        true_positive = cytology_result == ScreeningTestResult.POSITIVE or any(
            result == ScreeningTestResult.POSITIVE for result in hpv_result.values()
        )

        # apply cotest sensitivity and specificity
        if true_positive:
            cotest_result = (
                ScreeningTestResult.POSITIVE
                if self.model.rng.random() < self.params.sensitivity
                else ScreeningTestResult.NEGATIVE
            )
        else:
            cotest_result = (
                ScreeningTestResult.NEGATIVE
                if self.model.rng.random() < self.params.specificity
                else ScreeningTestResult.POSITIVE
            )

        return {
            "cotest": cotest_result,
            "cytology": cytology_result,
            "hpv": hpv_result,
            "hpv_16_18": any(
                hpv_result[strain] == ScreeningTestResult.POSITIVE
                for strain in [HpvStrain.SIXTEEN, HpvStrain.EIGHTEEN]
            ),
            "hr_hpv": any(
                hpv_result[strain] == ScreeningTestResult.POSITIVE
                for strain in [HpvStrain.HIGH_RISK]
            ),
            "lr_hpv": any(
                hpv_result[strain] == ScreeningTestResult.POSITIVE
                for strain in [HpvStrain.LOW_RISK]
            ),
        }


class CancerInspectionScreeningTest:
    def __init__(self, model):
        self.model = model
        self.params = model.params.screening.cancer_inspection

    def get_result(self, true_cancer_state: CancerState) -> ScreeningTestResult:
        """Return the screening test result given a woman's true cancer state.

        Properties of the test:
        - Detectable cancer states are REGIONAL and DISTANT.
        - If the woman has a detectable state and the test is sensitive, then return CANCER.
        - If the woman has a detectable state and the test isn't sensitive, then return NEGATIVE.
        - If the woman doesn't have a detectable state and the test is specific, then return NEGATIVE.
        - If the woman doesn't have a detectable state and the test isn't specific, then return CANCER.

        Raise a ValueError if true_cancer_state is DEAD.
        """

        if true_cancer_state == CancerState.DEAD:
            raise ValueError()
        elif true_cancer_state in [CancerState.NORMAL, CancerState.LOCAL]:
            if self.model.rng.random() > self.params.specificity:
                return ScreeningTestResult.CANCER
            else:
                return ScreeningTestResult.NEGATIVE
        else:
            if self.model.rng.random() < self.params.sensitivity:
                return ScreeningTestResult.CANCER
            else:
                return ScreeningTestResult.NEGATIVE


def is_due_for_screening(model, unique_id):
    """Return True if the woman is due for a screening test based on her current screening state,
        her screening history, and the screening interval guidelines.

    Requirements:
    - Women who have been diagnosed with cancer are no longer screened.
    - Women in the ROUTINE screening state aren't screened if they are younger or older than threshold ages.
    - Women who have been screened within the past N years aren't due to be screened again,
        where N is an interval that varies by screening state.
    - Women with HIV have a different screening interval.
        This interval will be used in place of her state-based interval if the HIV-specific interval is shorter.
    """

    if model.cancer_detection.values[unique_id] == CancerDetectionState.DETECTED:
        return False

    t1 = model.age < model.params.screening.age_routine_start
    t2 = model.age > model.params.screening.age_routine_end
    if model.screening_state.values[unique_id] == ScreeningState.ROUTINE and (t1 or t2):
        return False

    if model.screening_state.values[unique_id] == ScreeningState.ROUTINE:
        interval = model.params.screening.interval_routine
    elif model.screening_state.values[unique_id] == ScreeningState.RE_TEST:
        interval = model.params.screening.interval_re_test
    elif model.screening_state.values[unique_id] == ScreeningState.SURVEILLANCE:
        interval = model.params.screening.interval_surveillance
    else:
        raise NotImplementedError(
            f"Unexpected screening state: {model.screening_state.values[unique_id]}"
        )

    if unique_id in model.hiv_detected:
        interval = min(interval, model.params.screening.interval_hiv)
    if unique_id not in model.dicts.last_screen_age:
        return True
    if model.age - model.dicts.last_screen_age.get(unique_id, 0) >= interval:
        return True
    return False


def is_compliant_with_screening(model, unique_id):
    if model.screening_state.values[unique_id] == ScreeningState.ROUTINE:
        return model.compliant_routine_state.values[unique_id]
    elif model.screening_state.values[unique_id] == ScreeningState.SURVEILLANCE:
        return model.compliant_surveillance_state.values[unique_id]
    return True


class ScreeningProtocol:
    def __init__(
        self,
        model,
        params: ScreeningParameters,
        via_screening_test: ViaScreeningTest,
        dna_screening_test: DnaScreeningTest,
        cytology_screening_test: CytologyScreeningTest,
        cotest_screening_test: CoTestScreeningTest,
        cancer_inspection_screening_test: CancerInspectionScreeningTest,
    ):
        self.params = params
        self.dna_screening_test = dna_screening_test
        self.via_screening_test = via_screening_test
        self.cytology_screening_test = cytology_screening_test
        self.cancer_inspection_screening_test = cancer_inspection_screening_test
        self.model = model

    def apply(self):
        raise NotImplementedError("Must implement this method in a subclass")

    def get_via_result(self, unique_id):
        return self.via_screening_test.get_result(
            true_hpv_state=self.model.max_hpv_state.values[unique_id],
            true_cancer_state=self.model.cancer.values[unique_id],
        )

    def get_dna_result(self, unique_id):
        return self.dna_screening_test.get_result(
            true_hpv_states={
                strain: self.model.hpv_strains[strain].values[unique_id]
                for strain in HpvStrain
            }
        )

    def get_cytology_result(self, unique_id):
        return self.cytology_screening_test.get_result(
            true_cytology_state=self.model.cytology.values[unique_id],
        )

    def get_cotest_result(self, unique_id):
        return self.cotest_screening_test.get_result(
            true_cotest_state=self.model.cotest.values[unique_id],
        )

    def get_cancer_inspection_result(self, unique_id):
        return self.cancer_inspection_screening_test.get_result(
            true_cancer_state=self.model.cancer.values[unique_id],
        )


class NoScreeningProtocol(ScreeningProtocol):
    def __init__(self, *args, **kwargs):
        pass

    def is_due(self, *args, **kwargs):
        return False

    def apply(self, *args, **kwargs):
        pass


class ViaScreeningProtocol(ScreeningProtocol):
    def apply(self, unique_id=None):
        unique_ids = [unique_id]
        if unique_id is None:
            unique_ids = self.model.unique_ids[self.model.life.living]
        for unique_id in unique_ids:
            if not is_due_for_screening(self.model, unique_id):
                continue
            if not is_compliant_with_screening(self.model, unique_id):
                continue
            self.model.dicts.last_screen_age[unique_id] = self.model.age

            if (
                self.model.screening_state.values[unique_id]
                == ScreeningState.SURVEILLANCE
            ):
                event = Event.SURVEILLANCE_VIA
            else:
                event = Event.SCREENING_VIA

            self.model.events.record_event(
                (self.model.time, unique_id, event.value, self.params.via.cost)
            )

            result = self.get_via_result(unique_id)

            if result == ScreeningTestResult.NEGATIVE:
                self.model.screening_state.values[unique_id] = ScreeningState.ROUTINE
            elif result == ScreeningTestResult.POSITIVE:
                self.model.treat_cin(unique_id)
                self.model.screening_state.values[
                    unique_id
                ] = ScreeningState.SURVEILLANCE
            elif result == ScreeningTestResult.CANCER:
                self.model.detect_cancer(unique_id)
                self.model.screening_state.values[
                    unique_id
                ] = ScreeningState.SURVEILLANCE
            else:
                raise NotImplementedError(f"Unexpected screening test result: {result}")


class DnaThenTreatmentScreeningProtocol(ScreeningProtocol):
    def apply(self, unique_id=None):
        unique_ids = [unique_id]
        if unique_id is None:
            unique_ids = self.model.unique_ids[self.model.life.living]
        for unique_id in unique_ids:
            if not is_due_for_screening(self.model, unique_id):
                continue
            if not is_compliant_with_screening(self.model, unique_id):
                continue

            self.model.dicts.last_screen_age[unique_id] = self.model.age

            if (
                self.model.screening_state.values[unique_id]
                == ScreeningState.SURVEILLANCE
            ):
                event = Event.SURVEILLANCE_DNA
            else:
                event = Event.SCREENING_DNA
            self.model.events.record_event(
                (self.model.time, unique_id, event.value, self.params.dna.cost)
            )

            result = self.get_dna_result(unique_id)

            if all(
                result[strain] == ScreeningTestResult.NEGATIVE for strain in HpvStrain
            ):
                self.model.screening_state.values[unique_id] = ScreeningState.ROUTINE
            else:
                if (
                    self.model.screening_state.values[unique_id]
                    == ScreeningState.SURVEILLANCE
                ):
                    event2 = Event.SURVEILLANCE_CANCER_INSPECTION
                else:
                    event2 = Event.SCREENING_CANCER_INSPECTION

                self.model.events.record_event(
                    (
                        self.model.time,
                        unique_id,
                        event2.value,
                        self.params.cancer_inspection.cost,
                    )
                )

                result = self.get_cancer_inspection_result(unique_id)

                if result == ScreeningTestResult.NEGATIVE:
                    self.model.treat_cin(unique_id)
                    self.model.screening_state.values[
                        unique_id
                    ] = ScreeningState.SURVEILLANCE
                elif result == ScreeningTestResult.CANCER:
                    self.model.detect_cancer(unique_id)
                    self.model.screening_state.values[
                        unique_id
                    ] = ScreeningState.SURVEILLANCE
                else:
                    raise NotImplementedError(
                        f"Unexpected screening test result: {result}"
                    )


class DnaThenViaScreeningProtocol(ScreeningProtocol):
    def apply(self, unique_id=None):
        unique_ids = [unique_id]
        if unique_id is None:
            unique_ids = self.model.unique_ids[self.model.life.living]

        for unique_id in unique_ids:
            if not is_due_for_screening(self.model, unique_id):
                continue
            if not is_compliant_with_screening(self.model, unique_id):
                continue

            self.model.dicts.last_screen_age[unique_id] = self.model.age

            if (
                self.model.screening_state.values[unique_id]
                == ScreeningState.SURVEILLANCE
            ):
                event = Event.SURVEILLANCE_DNA
            else:
                event = Event.SCREENING_DNA

            self.model.events.record_event(
                (self.model.time, unique_id, event.value, self.params.dna.cost)
            )
            result = self.get_dna_result(unique_id)

            stp_pos = ScreeningTestResult.POSITIVE
            if all(
                result[strain] == ScreeningTestResult.NEGATIVE for strain in HpvStrain
            ):
                self.model.screening_state.values[unique_id] = ScreeningState.ROUTINE
            elif (result[HpvStrain.SIXTEEN] == stp_pos) or (
                result[HpvStrain.EIGHTEEN] == stp_pos
            ):
                if (
                    self.model.screening_state.values[unique_id]
                    == ScreeningState.SURVEILLANCE
                ):
                    event2 = Event.SURVEILLANCE_CANCER_INSPECTION
                else:
                    event2 = Event.SCREENING_CANCER_INSPECTION

                self.model.events.record_event(
                    (
                        self.model.time,
                        unique_id,
                        event2.value,
                        self.params.cancer_inspection.cost,
                    )
                )
                result = self.get_cancer_inspection_result(unique_id)

                if result == ScreeningTestResult.NEGATIVE:
                    self.model.treat_cin(unique_id)
                    self.model.screening_state.values[
                        unique_id
                    ] = ScreeningState.SURVEILLANCE
                elif result == ScreeningTestResult.CANCER:
                    self.model.detect_cancer(unique_id)
                    self.model.screening_state.values[
                        unique_id
                    ] = ScreeningState.SURVEILLANCE
                else:
                    raise NotImplementedError(
                        f"Unexpected screening test result: {result}"
                    )
            else:
                if (
                    self.model.screening_state.values[unique_id]
                    == ScreeningState.SURVEILLANCE
                ):
                    event2 = Event.SURVEILLANCE_VIA
                else:
                    event2 = Event.SCREENING_VIA

                self.model.events.record_event(
                    (self.model.time, unique_id, event2.value, self.params.via.cost)
                )
                result = self.get_via_result(unique_id)

                if result == ScreeningTestResult.NEGATIVE:
                    self.model.screening_state.values[
                        unique_id
                    ] = ScreeningState.RE_TEST
                elif result == ScreeningTestResult.POSITIVE:
                    self.model.treat_cin(unique_id)
                    self.model.screening_state.values[
                        unique_id
                    ] = ScreeningState.SURVEILLANCE
                elif result == ScreeningTestResult.CANCER:
                    self.model.detect_cancer(unique_id)
                    self.model.screening_state.values[
                        unique_id
                    ] = ScreeningState.SURVEILLANCE
                else:
                    raise NotImplementedError(
                        f"Unexpected screening test result: {result}"
                    )


class DnaThenTriageScreeningProtocol(ScreeningProtocol):
    def apply(self, unique_id=None):
        unique_ids = [unique_id]
        if unique_id is None:
            unique_ids = self.model.unique_ids[self.model.life.living]

        for unique_id in unique_ids:
            if not is_due_for_screening(self.model, unique_id):
                continue
            if not is_compliant_with_screening(self.model, unique_id):
                continue

            self.model.dicts.last_screen_age[unique_id] = self.model.age

            if (
                self.model.screening_state.values[unique_id]
                == ScreeningState.SURVEILLANCE
            ):
                event = Event.SURVEILLANCE_DNA
            else:
                event = Event.SCREENING_DNA

            self.model.events.record_event(
                (self.model.time, unique_id, event.value, self.params.dna.cost)
            )
            result = self.get_dna_result(unique_id)

            stp_pos = ScreeningTestResult.POSITIVE
            if all(
                result[strain] == ScreeningTestResult.NEGATIVE for strain in HpvStrain
            ):
                self.model.screening_state.values[unique_id] = ScreeningState.ROUTINE
            elif (result[HpvStrain.SIXTEEN] == stp_pos) or (
                result[HpvStrain.EIGHTEEN] == stp_pos
            ):
                if (
                    self.model.screening_state.values[unique_id]
                    == ScreeningState.SURVEILLANCE
                ):
                    event2 = Event.SURVEILLANCE_CANCER_INSPECTION
                else:
                    event2 = Event.SCREENING_CANCER_INSPECTION
                self.model.events.record_event(
                    (
                        self.model.time,
                        unique_id,
                        event2.value,
                        self.params.cancer_inspection.cost,
                    )
                )
                result = self.get_cancer_inspection_result(unique_id)

                if result == ScreeningTestResult.NEGATIVE:
                    self.model.treat_cin(unique_id)
                    self.model.screening_state.values[
                        unique_id
                    ] = ScreeningState.SURVEILLANCE
                elif result == ScreeningTestResult.CANCER:
                    self.model.detect_cancer(unique_id)
                    self.model.screening_state.values[
                        unique_id
                    ] = ScreeningState.SURVEILLANCE
                else:
                    raise NotImplementedError(
                        f"Unexpected screening test result: {result}"
                    )
            else:
                self.model.screening_state.values[unique_id] = ScreeningState.RE_TEST


class ScreeningStrategy:
    def __init__(
        self,
        model,
        params: ScreeningParameters,
        strategy_params: ScreeningStrategyParameters,
    ):
        self.model = model
        self.params = params
        self.strategy_params = strategy_params
        self.cytology_test = CytologyScreeningTest(model)
        self.dna_test = DnaScreeningTest(model)
        self.cotest = CoTestScreeningTest(model)

    def perform_screening(self, unique_id):
        age = self.model.age[unique_id]
        test = self.get_screening_test(age)

        if test == ScreeningTest.CYTOLOGY:
            return self.cytology_test.perform_test(unique_id)
        elif test == ScreeningTest.HPV:
            return self.hpv_test.perform_test(unique_id)
        elif test == ScreeningTest.COTEST:
            if 30 <= age <= 65:
                return self.cotest.perform_test(unique_id)
            else:
                # Fallback to cytology if age is not within cotest range
                return self.cytology_test.perform_test(unique_id)
        else:
            raise ValueError(f"Unknown screening test: {test}")

    def triage_results(self, unique_id, results):
        if "cotest" in results:
            return self.triage_cotest_results(unique_id, results)
        elif "cytology" in results:
            return self.triage_cytology_results(unique_id, results["cytology"])
        elif "hpv" in results:
            return self.triage_hpv_results(unique_id, results["hpv"])
        else:
            raise ValueError("Invalid screening results")

    def triage_cytology_results(self, unique_id, result):
        cytology_state = self.model.cervical_lesion.values[unique_id]
        if result == ScreeningTestResult.NEGATIVE:
            # 1. Negative (Repeat after 3 years)
            return ScreeningState.ROUTINE
        elif cytology_state == CervicalLesionState.ASCUS:
            # 2. ASCUS: followup with cotest in a year
            return ScreeningState.RE_TEST
        elif cytology_state in [
            CervicalLesionState.LSIL,
            CervicalLesionState.ASCH,
            CervicalLesionState.HSIL,
        ]:
            # 3. if LSIL, ASC-H, HSIL go for colposcopy
            return ScreeningState.COLPOSCOPY
        else:
            # Default case
            return ScreeningState.RE_TEST

    def triage_dna_results(self, unique_id, result):
        if all(r == ScreeningTestResult.NEGATIVE for r in result.values()):
            # 1. Negative (Repeat in 5 yrs.)
            return ScreeningState.ROUTINE
        elif (
            result[HpvStrain.SIXTEEN] == ScreeningTestResult.POSITIVE
            or result[HpvStrain.EIGHTEEN] == ScreeningTestResult.POSITIVE
        ):
            # 3. if HPV 16/18: Colposcopy
            return ScreeningState.COLPOSCOPY
        else:
            # 2. Hr HPV: If Other HrHPV: COtest in a year
            return ScreeningState.RE_TEST

    def triage_cotest_results(self, unique_id, results):
        cotest_result = results["cotest"]
        cytology_result = results["cytology"]
        hr_hpv_result = results["hr_hpv"]
        hpv_16_18_result = results["hpv_16_18"]

        if cotest_result == ScreeningTestResult.NEGATIVE:
            # 1. CoTest -ve (equivalent to HrHPV -ve, PAP -ve) (Repeat in 5 yrs.)
            return ScreeningState.ROUTINE
        else:
            # CoTest +ve, now we need to look at individual test results
            if hr_hpv_result and cytology_result == ScreeningTestResult.NEGATIVE:
                # 2. HrHPV +ve, PAP -ve; tested again in 1 year
                if hpv_16_18_result:
                    # b. HPV 16/18 +ve: Go for colposcopy
                    return ScreeningState.COLPOSCOPY
                else:
                    # a. -ve HPV 16/18 (Followup Cotest -1 yr.)
                    return ScreeningState.RE_TEST
            elif hr_hpv_result and cytology_result == ScreeningTestResult.POSITIVE:
                # 3. HrHPV +ve, PAP +ve: Go for colposcopy
                return ScreeningState.COLPOSCOPY
            elif not hr_hpv_result and cytology_result == ScreeningTestResult.POSITIVE:
                # 4. HrHPV -ve, PAP +ve (Follow up in 1 yr.)
                cytology_state = self.model.cervical_lesion.values[unique_id]
                if cytology_state in [
                    CervicalLesionState.ASCH,
                    CervicalLesionState.LSIL,
                    CervicalLesionState.HSIL,
                ]:
                    # a. ASC-H/LSIL/HSIL: Go for colposcopy
                    return ScreeningState.COLPOSCOPY
                elif cytology_state == CervicalLesionState.ASCUS:
                    # b. ASC-US Followup Cotest -1 yr.
                    return ScreeningState.RE_TEST
                else:
                    # For any other cytology result, default to re-test
                    return ScreeningState.RE_TEST
            else:
                # This case should not occur given our cotest logic, but included for completeness
                return ScreeningState.RE_TEST


"""class CytologyScreeningProtocol(ScreeningProtocol):
    def apply(self, unique_id=None):
        unique_ids = [unique_id]
        if unique_id is None:
            unique_ids = self.model.unique_ids[self.model.life.living]
        for unique_id in unique_ids:
            if not is_due_for_screening(self.model, unique_id):
                continue
            if not is_compliant_with_screening(self.model, unique_id):
                continue
            self.model.dicts.last_screen_age[unique_id] = self.model.age

            if (
                self.model.screening_state.values[unique_id]
                == ScreeningState.SURVEILLANCE
            ):
                event = Event.SURVEILLANCE_CYTOLOGY
            else:
                event = Event.SCREENING_CYTOLOGY

            self.model.events.record_event(
                (self.model.time, unique_id, event.value, self.params.cytology.cost)
            )

            result = self.get_cytology_result(unique_id)

            if result == ScreeningTestResult.NEGATIVE:
                self.model.screening_state.values[unique_id] = ScreeningState.ROUTINE
            elif result == ScreeningTestResult.POSITIVE:
                self.model.treat_cin(unique_id)
                self.model.screening_state.values[
                    unique_id
                ] = ScreeningState.SURVEILLANCE
            else:
                raise NotImplementedError(f"Unexpected screening test result: {result}")
"""

protocols = {
    "none": NoScreeningProtocol,
    "via": ViaScreeningProtocol,
    "dna_then_treatment": DnaThenTreatmentScreeningProtocol,
    "dna_then_via": DnaThenViaScreeningProtocol,
    "dna_then_triage": DnaThenTriageScreeningProtocol,
    "cytology": CytologyScreeningProtocol,
}
