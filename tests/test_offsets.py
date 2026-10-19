from pathlib import Path

import pandas as pd
import pytest

from surface_temperature import (
    MEAN_OF_DATASETS,
    Assessment,
    Baseline,
    compare_assessments,
    compute_offsets,
    load_assessments,
    load_baselines,
    rebase,
)

EARLY = Baseline("1850-1851", 1850, 1851, "early", "", "")
BRIDGE = Baseline("1852-1853", 1852, 1853, "bridge", "", "")


def annual(values: dict[str, dict[int, float]]) -> pd.DataFrame:
    rows = [
        (dataset, "GMST", "native", "central", year, value)
        for dataset, series in values.items()
        for year, value in series.items()
    ]
    return pd.DataFrame(
        rows, columns=["dataset", "quantity", "baseline", "statistic", "year", "value"]
    )


def test_bridged_offset_carries_the_covering_datasets_gap():
    frame = annual(
        {
            "a": {1850: 0.0, 1851: 0.0, 1852: 1.0, 1853: 1.0},
            "b": {1850: 1.0, 1851: 1.0, 1852: 3.0, 1853: 3.0},
            "late": {1852: 5.0, 1853: 5.0},
        }
    )
    offsets = compute_offsets(frame, [EARLY, BRIDGE], BRIDGE.id).set_index(["dataset", "baseline"])

    assert offsets.loc[("a", "1850-1851"), "method"] == "direct"
    # The mean bridge-minus-early gap of a and b is 1.5, so late sits 1.5 above its bridge mean.
    assert offsets.loc[("late", "1850-1851"), "method"] == "bridged"
    assert offsets.loc[("late", "1850-1851"), "offset"] == pytest.approx(5.0 - 1.5)
    assert offsets.loc[("late", "1850-1851"), "bridge_datasets"] == "a, b"


def test_rebase_shifts_every_statistic_by_the_offset():
    frame = annual({"a": {1850: 0.0, 1851: 2.0, 1852: 1.0, 1853: 1.0}})
    bounds = frame.assign(statistic="upper", value=frame["value"] + 0.5)
    both = pd.concat([frame, bounds])
    rebased = rebase(both, compute_offsets(both, [EARLY, BRIDGE], BRIDGE.id))
    early = rebased[(rebased["baseline"] == "1850-1851") & (rebased["year"] == 1851)]

    assert early.set_index("statistic")["value"].to_dict() == {
        "central": 1.0,
        "upper": 1.5,
    }


def test_dataset_missing_the_bridge_is_refused():
    frame = annual({"a": {1850: 0.0, 1851: 0.0, 1852: 1.0, 1853: 1.0}, "short": {1853: 1.0}})
    with pytest.raises(ValueError, match="bridge"):
        compute_offsets(frame, [EARLY, BRIDGE], BRIDGE.id)


def test_baselines_file_loads():
    baselines, bridge = load_baselines(Path(__file__).parent.parent / "baselines.yaml")
    assert bridge in {b.id for b in baselines}


def test_comparison_covers_each_dataset_and_their_mean():
    frame = annual({"a": {1850: 1.0, 1851: 1.0}, "b": {1850: 2.0, 1851: 2.0}})
    rebased = frame.assign(baseline="1850-1851")
    estimate = Assessment("AR", "ch", "url", "GMST", "1850-1851", "1850-1851", 1.4, (1.0, 1.6))
    comparison = compare_assessments(rebased, [estimate]).set_index("dataset")

    assert comparison["observed"].to_dict() == {"a": 1.0, "b": 2.0, MEAN_OF_DATASETS: 1.5}
    assert comparison["within_range"].to_dict() == {"a": True, "b": False, MEAN_OF_DATASETS: True}


def test_assessments_file_names_only_known_baselines():
    root = Path(__file__).parent.parent
    baselines, _ = load_baselines(root / "baselines.yaml")
    assert load_assessments(root / "assessments.yaml", baselines)
