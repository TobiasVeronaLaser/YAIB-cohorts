import pytest

from src import weavehr


@pytest.mark.parametrize(
    "family, expected",
    [
        ("eicu", "oxygen_saturation"),
        ("hirid", "oxygen_saturation"),
        ("aumc", "oxygen_saturation"),
        (
            "mimic-iv",
            ["oxygen_saturation", "arterial_oxygen_saturation"],
        ),
    ],
)
def test_o2sat_dataset_override(family, expected):
    sources = [
        src for src in weavehr.ext["sources"]
        if weavehr.source_family(src) == family
    ]
    assert sources, f"No configured source for {family}"

    for src in sources:
        cfg = weavehr._concept_cfg(src, "o2sat")
        assert cfg["weavehr"] == expected


def test_other_concepts_unchanged():
    for src in weavehr.ext["sources"]:
        cfg = weavehr._concept_cfg(src, "urine")
        assert cfg["weavehr"] == "urine_output"
        assert cfg["aggregate"] == "sum"


def test_original_configuration_not_mutated():
    before = weavehr.ext["concepts"]["o2sat"].copy()

    for src in weavehr.ext["sources"]:
        weavehr._concept_cfg(src, "o2sat")

    assert weavehr.ext["concepts"]["o2sat"] == before


@pytest.mark.parametrize(
    "src, general_values, arterial_values, expected, concepts",
    [
        (
            "hirid",
            [80.0, 100.0],
            [100.0],
            90.0,
            "oxygen_saturation",
        ),
        (
            "eicu-crd",
            [80.0, 100.0],
            [100.0],
            90.0,
            "oxygen_saturation",
        ),
        (
            "aumc",
            [80.0, 100.0],
            [100.0],
            90.0,
            "oxygen_saturation",
        ),
        (
            "mimic-iv",
            [80.0],
            [100.0],
            90.0,
            ["oxygen_saturation", "arterial_oxygen_saturation"],
        ),
    ],
)
def test_o2sat_hourly_aggregation(
    monkeypatch,
    src,
    general_values,
    arterial_values,
    expected,
    concepts,
):
    import polars as pl

    calls = []

    def fake_concept(dataset, selected, *, required=False):
        assert dataset == src
        calls.append(selected)

        names = [selected] if isinstance(selected, str) else selected
        values = []

        for name in names:
            if name == "oxygen_saturation":
                values.extend(general_values)
            elif name == "arterial_oxygen_saturation":
                values.extend(arterial_values)
            else:
                raise AssertionError(f"Unexpected concept: {name}")

        return pl.DataFrame(
            {
                "subject_id": [1] * len(values),
                "stay_id": [1] * len(values),
                "time": [0] * len(values),
                "numeric_value": values,
                "text_value": [None] * len(values),
            },
            schema_overrides={"text_value": pl.String},
        ).lazy()

    monkeypatch.setattr(weavehr, "_concept", fake_concept)
    monkeypatch.setattr(
        weavehr,
        "_in_stays",
        lambda rows, dataset, interval: rows,
    )

    result = weavehr._load(
        src, "o2sat", weavehr._interval(None)
    )

    assert result.height == 1
    assert result["stay_id"][0] == 1
    assert result["time"][0] == 0
    assert result["o2sat"][0] == pytest.approx(expected)
    assert calls == [concepts]
