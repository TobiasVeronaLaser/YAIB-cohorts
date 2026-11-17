"""ICU stays, ICU death and hospital IDs read by src/weavehr.py from synthetic WeavEHR projects.

Each test writes extraction output in WeavEHR's project layout
(<project>/datasets/extraction/data/<src>/<version>/<table>/<EVENT>.parquet) and reads it through the configured
sources of weavehr-extensions/config.yml.
"""
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src import weavehr
from src.weavehr import SOURCES, load_concepts, source_family, stay_windows

T0 = datetime(2150, 1, 1, 8)
MEDS = {"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.String, "numeric_value": pl.Float32,
        "text_value": pl.String}


def at(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Point the loader at a temporary project per source; returns a writer of extracted events."""
    monkeypatch.setitem(weavehr.conf, "weavehr_project_path", str(tmp_path / "{src}"))
    weavehr._stays.cache_clear()

    def write(src: str, version: str, events: str, rows: list[dict], extra: dict | None = None) -> None:
        path = tmp_path / src / "datasets/extraction/data" / src / version / f"{events}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = {**MEDS, **(extra or {})}
        pl.DataFrame([{col: row.get(col) for col in schema} for row in rows], schema=schema).write_parquet(path)

    yield write
    weavehr._stays.cache_clear()


def test_every_source_is_configured_and_ricu_variants_follow_their_dataset():
    for src in SOURCES:
        spec = weavehr.ext["sources"][src]
        assert {"family", "stay_id", "icu_admission", "icu_discharge", "death_icu"} <= set(spec), src
        assert spec["family"] in {"eicu", "mimic-iv", "hirid", "aumc"}, src
    for variant, dataset in [("eicu-crd-ricu", "eicu-crd"), ("eicu-demo-ricu", "eicu-crd"),
                             ("mimic-iv-ricu", "mimic-iv"), ("mimic-iv-demo-ricu", "mimic-iv")]:
        assert weavehr.ext["sources"][variant] == weavehr.ext["sources"][dataset]
        assert source_family(variant) == source_family(dataset)


def test_every_source_has_a_weavehr_dataset_config():
    loader = pytest.importorskip("weavehr.utils.loader")
    datasets = loader.bundled_config_path() / "datasets"
    assert [src for src in SOURCES if not (datasets / src).is_dir()] == []


EICU = ("eicu-demo-ricu", "2.0")
EICU_STAY = {"stay_id": pl.Int64}


def write_eicu(write):
    # subject 1: stay 10 (discharged alive), then stay 11 (died in the ICU); subject 2: stay 20, discharged alive
    # from the ICU and died later in the hospital
    write(*EICU, "patient/ICU_ADMISSION",
          [{"subject_id": 1, "stay_id": 10, "time": at(0), "hospital_id": 5},
           {"subject_id": 1, "stay_id": 11, "time": at(48), "hospital_id": 5},
           {"subject_id": 2, "stay_id": 20, "time": at(0), "hospital_id": 7}],
          {**EICU_STAY, "hospital_id": pl.Int64})
    write(*EICU, "patient/ICU_DISCHARGE",
          [{"subject_id": 1, "stay_id": 10, "time": at(24), "text_value": "Alive"},
           {"subject_id": 1, "stay_id": 11, "time": at(60), "text_value": "Expired"},
           {"subject_id": 2, "stay_id": 20, "time": at(10), "text_value": "Alive"}], EICU_STAY)
    write(*EICU, "patient/MEDS_DEATH",
          [{"subject_id": 1, "stay_id": 11, "time": at(60), "code": "MEDS_DEATH"},
           {"subject_id": 2, "stay_id": 20, "time": at(10), "code": "MEDS_DEATH"}], EICU_STAY)


def test_eicu_icu_death_is_the_icu_discharge_status_not_a_hospital_death(project):
    write_eicu(project)
    # subject 2's hospital death is at the ICU discharge (within the stay), but the ICU discharge status is Alive
    res = load_concepts("death_icu", EICU[0])
    assert res.to_dict("records") == [{"stay_id": 11, "time": 12, "death_icu": True}]


def test_eicu_hospital_id_is_the_extension_column_of_the_icu_admission(project):
    write_eicu(project)
    res = load_concepts("hospital_id", EICU[0])
    assert res.to_dict("records") == [{"stay_id": 10, "hospital_id": 5}, {"stay_id": 11, "hospital_id": 5},
                                      {"stay_id": 20, "hospital_id": 7}]


def test_eicu_stays_are_the_icu_admission_and_discharge(project):
    write_eicu(project)
    assert stay_windows(EICU[0]).to_dict("records") == [
        {"stay_id": 10, "start": 0, "end": 24}, {"stay_id": 11, "start": 0, "end": 12},
        {"stay_id": 20, "start": 0, "end": 10}]


HIRID = ("hirid", "1.1.1")


def write_hirid(write):
    write(*HIRID, "general/ICU_ADMISSION",
          [{"subject_id": 100, "time": at(0), "code": "admission"}, {"subject_id": 101, "time": at(0), "code": "admission"}])
    write(*HIRID, "general/MORTALITY",
          [{"subject_id": 100, "time": at(0), "code": "mortality", "text_value": "dead"},
           {"subject_id": 101, "time": at(0), "code": "mortality", "text_value": "alive"}])
    write(*HIRID, "observations/OBSERVATION",
          [{"subject_id": 100, "time": at(1), "code": "OBSERVATION//200//Heart rate//b/min", "numeric_value": 80},
           {"subject_id": 100, "time": at(3), "code": "OBSERVATION//110//ABPm//mmHg", "numeric_value": 60},
           {"subject_id": 100, "time": at(5), "code": "OBSERVATION//2000//other//unit", "numeric_value": 1},
           {"subject_id": 101, "time": at(2), "code": "OBSERVATION//200//Heart rate//b/min", "numeric_value": 70}])


def test_hirid_stays_end_at_the_last_observation(project):
    write_hirid(project)
    assert stay_windows(HIRID[0]).to_dict("records") == [
        {"stay_id": 100, "start": 0, "end": 5}, {"stay_id": 101, "start": 0, "end": 2}]
    assert load_concepts("los_icu", HIRID[0])["los_icu"].tolist() == pytest.approx([5 / 24, 2 / 24])


def test_hirid_icu_death_is_at_the_last_heart_rate_or_arterial_pressure(project):
    write_hirid(project)
    # discharge_status dead; the last observation of variable 110 or 200 is at 3h (the stay ends at 5h)
    assert load_concepts("death_icu", HIRID[0]).to_dict("records") == [{"stay_id": 100, "time": 3, "death_icu": True}]


def test_hirid_hospital_id_is_constant(project):
    write_hirid(project)
    assert load_concepts("hospital_id", HIRID[0])["hospital_id"].tolist() == [1, 1]


def test_a_source_without_hospital_id_raises(project):
    with pytest.raises(KeyError, match="hospital_id"):
        load_concepts("hospital_id", "aumc")
