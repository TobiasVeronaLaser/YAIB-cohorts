import os
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

from src import conf
from src.cohort import Cohort
from src.criteria import base_criteria
from src.steps import (
    InputStep, LoadStep, 
    CustomStep, RenameStep,
    Pipeline
)
from src.weavehr import SOURCES, stay_windows
from src.cohort_utils import (
    stop_window_at, make_grid_mapper, make_patient_mapper
)

outc_var = "los_icu"
static_vars = ["age", "sex", "height", "weight"]
dynamic_vars = ["alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir",
                  "bnd", "bun", "ca", "cai", "ck", "ckmb", "cl", "crea", "crp", 
                  "dbp", "fgn", "fio2", "glu", "hgb", "hr", "inr_pt", "k", "lact",
                  "lymph", "map", "mch", "mchc", "mcv", "methb", "mg", "na", "neut", 
                  "o2sat", "pco2", "ph", "phos", "plt", "po2", "ptt", "resp", "sbp", 
                  "temp", "tnt", "urine", "wbc"]

def create_los_task(args):
    print('Start creating the length of stay task.')
    print('   Preload variables')
    load_los = Pipeline('Load and process kidney function')
    load_los.add_step([
        LoadStep('los_icu_hours', args.src),
        RenameStep('los_icu_hours', outc_var)
    ])
    los = load_los.apply()

    load_static = LoadStep(static_vars, args.src, cache=True)
    load_dynamic = LoadStep(dynamic_vars, args.src, cache=True)

    print('   Define observation times')
    patients = stay_windows(args.src)
    patients = stop_window_at(patients, end=24*7)

    print('   Define exclusion criteria')
    excl1, excl2, excl3, excl4, excl5 = base_criteria(args.src, patients, load_dynamic)

    print('   Select cohort\n')
    cohort = Cohort(patients)
    cohort.add_criterion([excl1, excl2, excl3, excl4, excl5])
    print(cohort.criteria)
    patients, attrition = cohort.select()
    print('\n')

    print('   Load and format input data')
    def calculate_remaining_los(df):
        df['max_los'] = 7 * 24
        df["los_icu"] = df['los_icu'] - df['time']
        df["los_icu"] = df[['los_icu', 'max_los']].min(axis=1)
        return df.drop('max_los', axis=1)

    outc_formatting = Pipeline("Prepare length of stay")
    outc_formatting.add_step([
        InputStep(los), 
        CustomStep(make_grid_mapper(patients, match_time=False)),
        CustomStep(calculate_remaining_los),
        RenameStep(outc_var, 'label')
    ])
    outc = outc_formatting.apply()

    dyn_formatting = Pipeline("Prepare dynamic variables")
    dyn_formatting.add_step([
        load_dynamic,
        CustomStep(make_grid_mapper(patients, step_size=1))
    ])
    dyn = dyn_formatting.apply()

    sta_formatting = Pipeline("Prepare static variables")
    sta_formatting.add_step([
        load_static,
        CustomStep(make_patient_mapper(patients))
    ])
    sta = sta_formatting.apply()

    return (outc, dyn, sta), attrition


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='mimic-iv-demo', help='name of the WeavEHR dataset',
                        choices=SOURCES)
    args = parser.parse_known_args()[0]

    (outc, dyn, sta), attrition = create_los_task(args)

    save_dir = os.path.join(conf['out_dir'], 'los', args.src)
    os.makedirs(save_dir, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(outc), os.path.join(save_dir, 'outc.parquet'))
    pq.write_table(pa.Table.from_pandas(dyn), os.path.join(save_dir, 'dyn.parquet'))
    pq.write_table(pa.Table.from_pandas(sta), os.path.join(save_dir, 'sta.parquet'))

    attrition.to_csv(os.path.join(save_dir, 'attrition.csv'))
