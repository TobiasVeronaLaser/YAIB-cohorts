"""Exclusion criteria shared by every task (YAIB's base cohort, R/base_cohort.R exclusions 1-5)."""
from typing import List

import pandas as pd

from .cohort import SelectionCriterion
from .cohort_utils import longest_rle, make_grid_mapper, n_obs_per_row
from .steps import AggStep, CustomStep, FilterStep, InputStep, LoadStep, Step, TransformStep


def base_criteria(src: str, windows: pd.DataFrame, load_dynamic: Step) -> List[SelectionCriterion]:
    """The five base exclusions.

    Args:
        src: WeavEHR dataset (`--src`).
        windows: observation windows of all stays (`stay_windows`, capped at 7 days).
        load_dynamic: the task's (cached) load step of the dynamic variables.
    """
    excl1 = SelectionCriterion('Invalid length of stay')
    excl1.add_step([
        InputStep(windows),
        FilterStep('end', lambda x: x < 0)
    ])

    # completed hours in integer arithmetic; `los_icu < 6 / 24` in float days can misjudge a stay of exactly 6h
    excl2 = SelectionCriterion('Length of stay < 6h')
    excl2.add_step([
        LoadStep('los_icu_hours', src),
        FilterStep('los_icu_hours', lambda x: x < 6)
    ])

    excl3 = SelectionCriterion('Less than 4 hours with any measurement')
    excl3.add_step([
        load_dynamic,
        AggStep('stay_id', 'count'),
        FilterStep('time', lambda x: x < 4)
    ])

    excl4 = SelectionCriterion('More than 12 hour gap between measurements')
    excl4.add_step([
        load_dynamic,
        CustomStep(make_grid_mapper(windows, step_size=1)),
        CustomStep(n_obs_per_row),
        TransformStep('n', lambda x: x > 0),
        AggStep('stay_id', longest_rle, 'n'),
        FilterStep('n', lambda x: x > 12)
    ])

    excl5 = SelectionCriterion('Aged < 18 years')
    excl5.add_step([
        LoadStep('age', src),
        FilterStep('age', lambda x: x < 18)
    ])

    return [excl1, excl2, excl3, excl4, excl5]
