## Replicate the cohorts from Python:

The Python scripts derive the cohorts with [WeavEHR](https://github.com/aidh-ms/WeavEHR), a Python framework that harmonises ICU datasets into the MEDS format. They do not need R; the R scripts in [../R](../R/README.md) derive the cohorts with `ricu`.

The following instructions assume that this folder is the current working directory in your Python session.


### Step 0: Environment and Prerequisites

WeavEHR requires Python >= 3.13. To create the environment with conda, run:

```bash
conda env create -f environment.yml
conda activate yaib-cohorts
```

[environment.yml](environment.yml) installs the WeavEHR version that the cohorts were checked against. Alternatively, install `pandas`, `numpy`, `polars`, `pyyaml`, `pyarrow`, `scikit-learn` and that WeavEHR version with pip.

> Note: the Docker image in [../docker](../docker) is set up for `ricu` (R and Python 3.10) and does not run these scripts.

### Step 1: Build the WeavEHR project of a dataset

The cohorts are read from a WeavEHR project per dataset, the counterpart of importing a dataset into `ricu`. Configured datasets (`--src`, WeavEHR version): MIMIC-IV (`mimic-iv`, 2.2 or 3.1) and its openly available [demo](https://physionet.org/content/mimic-iv-demo/) (`mimic-iv-demo`, 2.2), eICU-CRD (`eicu-crd`, 2.0) and its [demo](https://physionet.org/content/eicu-crd-demo/) (`eicu-demo`, 2.0), HiRID (`hirid`, 1.1.1) and AmsterdamUMCdb (`aumc`, 1.5.0, the OMOP export).

To compare the cohorts with those extracted with `ricu`, build the ricu-compatible datasets `mimic-iv-ricu` (2.2 or 3.1), `mimic-iv-demo-ricu` (2.2), `eicu-crd-ricu` (2.0) and `eicu-demo-ricu` (2.0) from the same raw data. They are WeavEHR datasets that extend the dataset with ricu 0.6.3's mappings (with YAIB's `ricu-extensions`) where WeavEHR intentionally differs from ricu, e.g., magnesium in eICU or item 226253 in MIMIC-IV's oxygen saturation; their READMEs in WeavEHR (`configs/datasets/*-ricu/README.md`) list them. Each dataset gets its own project directory, so a `*-ricu` project does not replace the dataset's project.

`weavehr_project_path` in [config.yaml](../config.yaml) is where the WeavEHR project of each dataset (`{src}`) is kept. It contains patient-level data, so keep it outside of this repository. Build the project from the raw data files of the dataset:

```bash
python weavehr_build.py --src mimic-iv-demo --version 2.2 --data-path /path/to/physionet.org/files/mimic-iv-demo/2.2
python weavehr_build.py --src eicu-demo --version 2.0 --data-path /path/to/physionet.org/files/eicu-crd-demo/2.0.1
python weavehr_build.py --src eicu-demo-ricu --version 2.0 --data-path /path/to/physionet.org/files/eicu-crd-demo/2.0.1
```

The dataset and concept configs are WeavEHR's own; this repository adds none (see [Clinical concepts](#clinical-concepts)).


### Step 2: Generate the Cohorts

This repository currently allows for the extraction of five prediction tasks:

Classification:

1. ICU mortality after 24 hours: [mortality.py](mortality.py)
2. Acute Kidney Injury within the next 6 hours: [aki.py](aki.py)
3. Sepsis within the next 6 hours: [sepsis.py](sepsis.py)

Regression:

4. Kidney function on the second day of ICU admission: [kidney_function.py](kidney_function.py)
5. Remaining length of stay: [los.py](los.py)


Data for a single task like mortality from a single dataset can be extracted via:
```bash
python mortality.py --src mimic-iv-demo
```

where `mortality.py` should be replaced with the task file of interest and `mimic-iv-demo` with the dataset of interest (one of the `sources` in [weavehr-extensions/config.yml](../weavehr-extensions/config.yml): `mimic-iv`, `mimic-iv-demo`, `eicu-crd`, `eicu-demo`, their `*-ricu` datasets, `hirid` and `aumc`). Dataset-specific criteria (e.g., the hospital-level AKI and sepsis prevalence in eICU) follow the dataset `family`, so they also apply to the demo and the `*-ricu` datasets.

The output directory for the extracted data can also be set in [config.yaml](../config.yaml).


### Clinical concepts

The clinical concepts and dataset configs are those of WeavEHR, including the items that YAIB's `ricu-extensions` add to ricu (e.g., non-invasive blood pressure in eICU, MIMIC-IV lab items 53085, 53086 and 53089); they were previously added here by `weavehr-extensions/datasets/` and are now part of WeavEHR. [weavehr-extensions/config.yml](../weavehr-extensions/config.yml) maps the YAIB concepts to WeavEHR concepts, sets how each is aggregated per time bin (default: median; urine output: sum), and configures per dataset (`sources`) the ICU stays, ICU death and hospital in the WeavEHR extraction output.

[src/weavehr.py](src/weavehr.py) takes the ICU stays from the admission and discharge events of the WeavEHR project: the earliest admission and the latest discharge event of each stay. HiRID records no discharge, so, as in ricu, its stays end at their last observation. A time-series concept keeps the values recorded within an ICU stay of the patient, from admission to discharge, so values recorded before ICU admission are not used. A static concept (age, sex, height, weight) takes the last value recorded up to the end of the stay. The outcomes are WeavEHR's `aki` (the time bins with a KDIGO stage of 1 or more) and `sepsis3` (`sep3`, the first onset within the stay), ICU death within the stay (`death_icu`, see below), the length of the stay (`los_icu`) and the hospital of each stay (`hospital_id`; in eICU from `hospitalid`, otherwise 1, as in `ricu-extensions`).

Concepts are aggregated per hour by default. `load_concepts` and `stay_windows` in [src/weavehr.py](src/weavehr.py) also take another bin width, e.g., `LoadStep('hr', args.src, interval='30min')` or `interval='2h'`; `time` is then the start of each bin in hours since ICU admission (fractional for bins that are not whole hours). The task scripts are written for hourly bins (an hourly grid, and windows and criteria in hours), so they use the default.


### Differences from the cohorts extracted with ricu

The cohorts can be compared with those extracted with `ricu` using `weavehr-yaib-compare-cohorts` from [weavehr-yaib](https://github.com/TobiasVeronaLaser/weavehr-yaib). On the MIMIC-IV demo, the dynamic variables within the ICU stays agree with those extracted with ricu, apart from ricu's value ranges, which are not applied. Known differences:

- Value ranges: ricu drops values outside a range per concept (e.g., PaO2 below 40 mmHg); here all values of the WeavEHR concepts are used.
- Values before ICU admission: ricu attaches values to ICU stays through the IDs of their source table, so it also uses values recorded before ICU admission (e.g., labs earlier in the same hospital admission, at negative times). Here only values within the stay are used. This affects the criteria that look before admission: "Less than 4 hours with any measurement" (ricu counts hours before admission), "Baseline creatinine > 4" (AKI) and "onset before 6h" (AKI and sepsis), which does not see an onset before ICU admission.
- AKI is WeavEHR's KDIGO staging, graded at each creatinine and urine output record, with the stage of one criterion carried to the records of the other; ricu-extensions grades creatinine and urine output only at their own records, with a urine rate per hour.
- Sepsis is WeavEHR's `sepsis3`: suspected infection from antibiotics and body fluid sampling, and WeavEHR's SOFA score; ricu-extensions' `sep3_alt` requires antibiotics to be continued (`abx_cont`).
- Age (MIMIC-IV): WeavEHR's age is the age at each hospital admission, capped at 90; ricu uses the patient's `anchor_age`.
- Height and weight can come from before the stay (the last value up to its end); ricu uses the values attached to the stay.
- ICU death (`death_icu`): in eICU, as in ricu-extensions, an ICU discharge with status "Expired" (not a later death in the hospital); in HiRID, as in ricu-extensions, discharge status "dead" at the last heart rate or invasive mean arterial pressure observation. In MIMIC-IV, ricu-extensions counts in-hospital deaths whose last ward before discharge was an ICU; here a death (`MEDS_DEATH`) recorded within the stay; on the MIMIC-IV demo both give the same stays and times. In AmsterdamUMCdb, a `MEDS_DEATH` within the stay, which has not been checked against ricu-extensions (discharge destination "Overleden").
- On the demos, the `*-ricu` datasets give the same concepts as the former `weavehr-extensions/datasets/` overrides, except blood pressure in eICU (values outside ricu's limits, 0-300 and 0-200 mmHg, are dropped, as ricu does) and respiratory rate in the eICU demo (monitor values only, as in ricu-extensions for the demo; the former overrides also took the nurse-charted rates).
- AmsterdamUMCdb and HiRID have not been checked with this way of loading the concepts (there is no public demo of either). The sepsis task passes `si_mode` for eICU and HiRID, which `load_concepts` does not accept.
