"""ICU stays and clinical concepts for the cohort scripts, loaded from a WeavEHR project built with `weavehr_build.py`.

Returns tables in the form of those returned by `ricu::stay_windows()` and `ricu::load_concepts()`: one row per ICU
stay (`stay_id`) and time bin, or one row per stay for static concepts. Time bins are one hour unless `interval` is
given (e.g., "30min" or "2h"); `time` is the start of the bin in hours since ICU admission.

Time series keep the WeavEHR concept rows recorded during an ICU stay of the patient, aggregated per time bin; static
concepts take the last value recorded up to the end of the stay. Concepts are configured in
../weavehr-extensions/config.yml.
"""
from functools import cache
from pathlib import Path
from typing import List

import pandas as pd
import polars as pl
import yaml

from . import conf

with open("../weavehr-extensions/config.yml") as stream:  # this path is relative to the calling environment
    ext = yaml.safe_load(stream)

SOURCES = list(ext["sources"])
"""The WeavEHR datasets (`--src`) the cohorts can be extracted from."""


def source_family(src: str) -> str:
    """Dataset family of `src` (e.g., eicu for eicu-demo-ricu), by which the task scripts choose dataset criteria."""
    return ext["sources"][src]["family"]


HOUR = pd.Timedelta(hours=1)
AGGREGATE = {"median": pl.Expr.median, "mean": pl.Expr.mean, "sum": pl.Expr.sum, "min": pl.Expr.min, "max": pl.Expr.max}


def _interval(interval) -> pd.Timedelta:
    """Width of the time bins, a duration such as "30min" or "2h"; one hour by default."""
    res = pd.Timedelta(interval or HOUR)
    if isinstance(interval, (int, float)) or res <= pd.Timedelta(0):
        raise ValueError(f"interval must be a positive duration such as '2h', got {interval!r}.")
    return res


def _bin(col: str, interval: pd.Timedelta) -> pl.Expr:
    """Number of the time bin of `col` since ICU admission (`intime`)."""
    return (pl.col(col) - pl.col("intime")).dt.total_microseconds() // (interval // pd.Timedelta(microseconds=1))


def _hours(bins: pl.Expr, interval: pd.Timedelta) -> pl.Expr:
    """Start of the time bins numbered `bins` in hours; whole hours if the interval is."""
    return bins * (interval // HOUR) if interval % HOUR == pd.Timedelta(0) else bins * (interval / HOUR)


def _files(src: str, pattern: str) -> List[Path]:
    return sorted(Path(conf["weavehr_project_path"].format(src=src), "datasets").glob(pattern))


def _extracted(src: str, events: str, *columns: str) -> pl.LazyFrame:
    """Extracted events `<table>/<event>` of a dataset (`*` matches any table)."""
    files = _files(src, f"extraction/data/{src}/*/{events}.parquet")
    if not files:
        raise FileNotFoundError(f"No {events} events of {src}; build the project with weavehr_build.py.")
    return pl.concat([pl.scan_parquet(f).select(*columns) for f in files])


def _with_stay_id(src: str, events: str, *columns: str) -> pl.LazyFrame:
    """Extracted events with the configured ICU stay column as `stay_id` (the subject where a subject is one stay)."""
    stay = ext["sources"][src]["stay_id"]
    if stay == "subject_id":
        return _extracted(src, events, "subject_id", *columns).with_columns(pl.col("subject_id").alias("stay_id"))
    return _extracted(src, events, "subject_id", stay, *columns).rename({stay: "stay_id"})


@cache
def _stays(src: str) -> pl.DataFrame:
    """ICU stays: `stay_id`, `subject_id`, admission (`intime`) and discharge (`outtime`) time.

    The admission is the earliest `icu_admission` event of a stay and the discharge its latest `icu_discharge` event,
    which is the stay's last observation for HiRID (as in ricu).
    """
    spec = ext["sources"][src]

    def events(name: str, time: str, aggregate) -> pl.LazyFrame:
        return (_with_stay_id(src, spec[name], "time").group_by("subject_id", "stay_id")
                .agg(aggregate(pl.col("time")).alias(time)))

    stays = events("icu_admission", "intime", pl.Expr.min).join(
        events("icu_discharge", "outtime", pl.Expr.max), on=["subject_id", "stay_id"])
    return stays.collect().sort("stay_id")


def stay_windows(src: str, interval=None) -> pd.DataFrame:
    """Start of the first (`start`) and last (`end`) time bin of all ICU stays, in hours since ICU admission."""
    interval = _interval(interval)
    return _stays(src).select("stay_id", _hours(pl.lit(0), interval).alias("start"),
                              _hours(_bin("outtime", interval), interval).alias("end")).to_pandas()


def _concept(src: str, weavehr: str | List[str], *, required: bool = False) -> pl.LazyFrame:
    """MEDS rows of a WeavEHR concept, or of several pooled."""
    files = [f for name in ([weavehr] if isinstance(weavehr, str) else weavehr)
             for f in _files(src, f"concept/data/{name}/*/{src}/*.parquet")]
    if not files:
        if required:
            raise FileNotFoundError(
                f"Missing required WeavEHR concept {weavehr} for {src}."
            )
        print(f"WARNING: no output of WeavEHR concept {weavehr} for {src}.")
        return pl.LazyFrame(schema={"subject_id": pl.Int64, "time": pl.Datetime("us"), "numeric_value": pl.Float64,
                                    "text_value": pl.String})
    # WeavEHR stores float32 values; their shortest decimal is the value recorded in the source
    value = pl.col("numeric_value").cast(pl.Float32).cast(pl.String).cast(pl.Float64)
    return pl.concat([pl.scan_parquet(f).select("subject_id", "time", value, "text_value") for f in files])


def _in_stays(rows: pl.LazyFrame, src: str, interval: pd.Timedelta) -> pl.LazyFrame:
    """Rows recorded during an ICU stay of their patient, with `stay_id` and their time bin as `time`."""
    return (rows.join(_stays(src).lazy(), on="subject_id")
            .filter(pl.col("time").is_between("intime", "outtime"))
            .with_columns(_bin("time", interval).alias("time")))


def _load(src: str, name: str, interval: pd.Timedelta) -> pl.DataFrame:
    """Concept `name` of weavehr-extensions/config.yml: the last value up to the end of each stay (static), or the
    values during each stay aggregated per time bin."""
    cfg = ext["concepts"][name]
    rows = _concept(src, cfg["weavehr"]).with_columns(
        pl.col("text_value" if cfg.get("text") else "numeric_value").alias(name)).drop_nulls(name)
    if cfg.get("static"):
        return (rows.join(_stays(src).lazy(), on="subject_id").filter(pl.col("time") <= pl.col("outtime"))
                .group_by("stay_id").agg(pl.col(name).sort_by("time", maintain_order=True).last())
                .collect().sort("stay_id"))
    aggregate = AGGREGATE[cfg.get("aggregate", "median")]
    return (_in_stays(rows, src, interval).group_by("stay_id", "time").agg(aggregate(pl.col(name)))
            .collect().sort("stay_id", "time"))


def _events(src: str, name: str, rows: pl.LazyFrame, interval: pd.Timedelta, first: bool = False) -> pl.DataFrame:
    """Outcome `name`: true in the time bins with an event during the stay, or only in the first (`first`)."""
    res = _in_stays(rows, src, interval).select("stay_id", "time")
    res = res.group_by("stay_id").agg(pl.col("time").min()) if first else res.unique()
    return res.with_columns(pl.lit(True).alias(name)).collect().sort("stay_id", "time")


def _death_icu(src: str, interval: pd.Timedelta) -> pl.DataFrame:
    """ICU death as configured per dataset (`death_icu` in weavehr-extensions/config.yml), during the stay."""
    spec = ext["sources"][src]["death_icu"]
    rows = _extracted(src, spec["events"], "subject_id", "time", "text_value")
    if "text_value" in spec:
        rows = rows.filter(pl.col("text_value") == spec["text_value"])
    if "time_from" in spec:
        last = (_with_stay_id(src, spec["time_from"]["events"], "time", "code")
                .filter(pl.col("code").str.contains(spec["time_from"]["code"]))
                .group_by("subject_id", "stay_id").agg(pl.col("time").max()))
        rows = rows.select("subject_id").unique().join(last, on="subject_id").select("subject_id", "time")
    return _events(src, "death_icu", rows.select("subject_id", "time"), interval)


def _hospital_id(src: str, interval: pd.Timedelta) -> pl.DataFrame:
    """Hospital of each stay (`hospital_id` in weavehr-extensions/config.yml): an extension column of an event of the
    stay, or a constant."""
    spec = ext["sources"][src].get("hospital_id")
    if spec is None:
        raise KeyError(f"No hospital_id is configured for {src} in weavehr-extensions/config.yml.")
    stays = _stays(src).select("stay_id")
    if not isinstance(spec, str):
        return stays.with_columns(pl.lit(spec).alias("hospital_id"))
    events, column = spec.rsplit("/", 1)
    ids = _with_stay_id(src, events, column).select("stay_id", pl.col(column).alias("hospital_id")).unique()
    return stays.join(ids.collect(), on="stay_id", how="left").sort("stay_id")


OUTCOMES = {
    "los_icu": lambda src, interval: _stays(src).select(
        "stay_id", ((pl.col("outtime") - pl.col("intime")).dt.total_seconds() / 86400).alias("los_icu")),
    "death_icu": _death_icu,
    "hospital_id": _hospital_id,
    "aki": lambda src, interval: _events(src, "aki", _concept(src, "aki", required=True).filter(pl.col("numeric_value") >= 1), interval),
    "sep3": lambda src, interval: _events(src, "sep3", _concept(src, "sepsis3", required=True), interval, first=True),
}


def load_concepts(concepts: str | List[str], src: str, interval=None, *, si_mode=None) -> pd.DataFrame:
    """Load one or more concepts (time bins of `interval`, one hour by default), merged on `stay_id` and `time`.

    Outcomes: `los_icu` (days), `death_icu` (an ICU death during the stay, as configured per dataset), `hospital_id`
    (per stay), `aki` (WeavEHR's `aki`, KDIGO stage 1 or more) and `sep3` (the first onset of WeavEHR's `sepsis3`
    during the stay).
    """
    names = [concepts] if isinstance(concepts, str) else list(concepts)

    if si_mode is not None:
        if names != ["sep3"] or source_family(src) not in ("eicu", "hirid") or si_mode != "abx":
            raise ValueError(
                f"Unsupported si_mode={si_mode!r} for concepts={names}, src={src}"
            )
        # Equivalence to ricu's sep3_alt is NOT established.

    interval = _interval(interval)
    tables = [OUTCOMES[name](src, interval) if name in OUTCOMES else _load(src, name, interval) for name in names]
    res = tables[0]
    for table in tables[1:]:
        res = res.join(table, on=[c for c in ("stay_id", "time") if c in res.columns and c in table.columns],
                       how="full", coalesce=True)
    keys = [col for col in ("stay_id", "time") if col in res.columns]
    res = res.sort(keys).select(*keys, *names)
    if "time" in res.columns:
        res = res.with_columns(_hours(pl.col("time"), interval))
    return res.to_pandas()
