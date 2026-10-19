"""Parse the upstream global series and rebase them onto common baselines."""

import calendar
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

MONTHS = list(calendar.month_abbr)[1:]
STATISTICS = ("central", "lower", "upper")
# Rounded so the parquet bytes, and so the edition, only move when a value does.
DECIMALS = 4

Parser = Callable[[Mapping[str, Path]], tuple[pd.DataFrame, pd.DataFrame]]


@dataclass(frozen=True)
class Dataset:
    """One upstream product and how to read its global series."""

    id: str
    name: str
    producer: str
    quantity: str
    native_baseline: str
    inputs: tuple[str, ...]
    parse: Parser
    homepage: str
    licence: str
    uncertainty: str
    notes: str = ""
    attribution: tuple[str, ...] = ()


@dataclass(frozen=True)
class Baseline:
    """A reference period, inclusive of both end years."""

    id: str
    start: int
    end: int
    name: str
    use: str
    reference: str
    reference_url: str | None = None
    assessed: tuple[dict, ...] = field(default_factory=tuple)


def load_baselines(path: Path) -> tuple[list[Baseline], str]:
    """Read the baseline definitions and the bridge period from ``baselines.yaml``."""
    config = yaml.safe_load(path.read_text())
    baselines = [
        Baseline(**{**entry, "assessed": tuple(entry.get("assessed", ()))})
        for entry in config["baselines"]
    ]
    ids = [b.id for b in baselines]
    if len(set(ids)) != len(ids):
        raise ValueError(f"baseline ids repeat: {ids}")
    for b in baselines:
        if b.id != f"{b.start}-{b.end}":
            raise ValueError(f"baseline {b.id} does not match its years {b.start}-{b.end}")
    if config["bridge"] not in ids:
        raise ValueError(f"bridge {config['bridge']} is not a declared baseline")
    return baselines, config["bridge"]


def _monthly_from_wide(frame: pd.DataFrame, statistic: str = "central") -> pd.DataFrame:
    """Melt a year by month-name table into long monthly rows."""
    long = frame.melt(id_vars="year", value_vars=MONTHS, var_name="month", value_name="value")
    long["month"] = long["month"].map(MONTHS.index) + 1
    long["statistic"] = statistic
    return long


def _bounds(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Melt central, lower and upper columns into long statistic rows."""
    present = [s for s in STATISTICS if s in frame]
    return frame.melt(id_vars=keys, value_vars=present, var_name="statistic", value_name="value")


def _read_whitespace(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None)


def parse_hadcrut(paths: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the HadCRUT5 summary series, whose bounds are the 2.5% and 97.5% limits."""
    names = {"Anomaly (deg C)": "central"}
    names |= {
        "Lower confidence limit (2.5%)": "lower",
        "Upper confidence limit (97.5%)": "upper",
    }
    monthly = pd.read_csv(paths["hadcrut5_monthly"]).rename(columns=names)
    monthly[["year", "month"]] = monthly["Time"].str.split("-", expand=True).astype(int)
    annual = pd.read_csv(paths["hadcrut5_annual"]).rename(columns=names | {"Time": "year"})
    return _bounds(monthly, ["year", "month"]), _bounds(annual, ["year"])


def parse_noaa(paths: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read NOAAGlobalTemp, whose error variance columns are unfilled in the public files."""
    monthly = _read_whitespace(paths["noaaglobaltemp_monthly"]).iloc[:, :3]
    monthly.columns = ["year", "month", "central"]
    annual = _read_whitespace(paths["noaaglobaltemp_annual"]).iloc[:, :2]
    annual.columns = ["year", "central"]
    return _bounds(monthly, ["year", "month"]), _bounds(annual, ["year"])


def parse_gistemp(paths: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the GISTEMP land-ocean index, taking J-D as the annual mean."""
    table = pd.read_csv(paths["gistemp"], skiprows=1, na_values=["***"])
    table = table.rename(columns={"Year": "year"})
    annual = table[["year", "J-D"]].rename(columns={"J-D": "value"}).assign(statistic="central")
    return _monthly_from_wide(table), annual


def parse_era5(paths: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average the ERA5 daily series to complete months and complete years."""
    daily = pd.read_csv(paths["era5_daily"], comment="#", parse_dates=["date"])
    daily["year"], daily["month"] = daily["date"].dt.year, daily["date"].dt.month
    by_month = daily.groupby(["year", "month"])["ano_91-20"].agg(["mean", "size"]).reset_index()
    days = [calendar.monthrange(y, m)[1] for y, m in zip(by_month["year"], by_month["month"])]
    monthly = by_month[by_month["size"] == days]
    by_year = daily.groupby("year")["ano_91-20"].agg(["mean", "size"]).reset_index()
    annual = by_year[
        by_year["size"] == [366 if calendar.isleap(y) else 365 for y in by_year["year"]]
    ]
    return (
        monthly.rename(columns={"mean": "value"})[["year", "month", "value"]].assign(
            statistic="central"
        ),
        annual.rename(columns={"mean": "value"})[["year", "value"]].assign(statistic="central"),
    )


def _jma_number(column: pd.Series) -> pd.Series:
    # A trailing * marks a preliminary month.
    return pd.to_numeric(column.str.rstrip("*").str.lstrip("+"), errors="coerce")


def parse_jma(paths: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the JMA land-ocean series."""
    monthly = pd.read_csv(paths["jma_monthly"], dtype=str).rename(columns={"Year": "year"})
    monthly[MONTHS] = monthly[MONTHS].apply(_jma_number)
    monthly["year"] = monthly["year"].astype(int)
    annual = pd.read_csv(paths["jma_annual"], dtype=str).rename(columns={"Year": "year"})
    annual = pd.DataFrame(
        {"year": annual["year"].astype(int), "value": _jma_number(annual["Global"])}
    ).assign(statistic="central")
    return _monthly_from_wide(monthly), annual


DATASETS = (
    Dataset(
        id="HadCRUT5",
        name="HadCRUT5 analysis",
        producer="Met Office Hadley Centre and Climatic Research Unit, University of East Anglia",
        quantity="GMST",
        native_baseline="1961-1990",
        inputs=("hadcrut5_monthly", "hadcrut5_annual"),
        parse=parse_hadcrut,
        homepage="https://www.metoffice.gov.uk/hadobs/hadcrut5/",
        licence="Open Government Licence v3",
        attribution=(
            "HadCRUT.5.2.0.0 data were obtained from http://www.metoffice.gov.uk/hadobs/hadcrut5",
            "on the retrieval date above and are © British Crown Copyright, Met Office 2026,",
            "provided under an Open Government License,",
            "http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/.",
        ),
        uncertainty="2.5% and 97.5% limits of the 200-member ensemble plus coverage uncertainty",
    ),
    Dataset(
        id="NOAAGlobalTemp",
        name="NOAAGlobalTemp",
        producer="NOAA National Centers for Environmental Information",
        quantity="GMST",
        native_baseline="1991-2020",
        inputs=("noaaglobaltemp_monthly", "noaaglobaltemp_annual"),
        parse=parse_noaa,
        homepage="https://www.ncei.noaa.gov/products/land-based-station/noaa-global-temp",
        licence="US Government work, public domain",
        uncertainty="None published in the global series files",
    ),
    Dataset(
        id="GISTEMP",
        name="NASA GISTEMP v4 land-ocean temperature index",
        producer="NASA Goddard Institute for Space Studies",
        quantity="GMST",
        native_baseline="1951-1980",
        inputs=("gistemp",),
        parse=parse_gistemp,
        homepage="https://data.giss.nasa.gov/gistemp/",
        licence="US Government work, public domain",
        uncertainty="None published in the global series file",
    ),
    Dataset(
        id="ERA5",
        name="ERA5 global mean 2m temperature (Climate Pulse)",
        producer="Copernicus Climate Change Service / ECMWF",
        quantity="GSAT",
        native_baseline="1991-2020",
        inputs=("era5_daily",),
        parse=parse_era5,
        homepage="https://pulse.climate.copernicus.eu/",
        licence="CC-BY-4.0",
        attribution=(
            "ERA5 contains modified Copernicus Climate Change Service information.",
            "Neither the European Commission nor ECMWF is responsible for any use made of it.",
        ),
        uncertainty="None",
        notes=(
            "Reanalysis air temperature everywhere, so GSAT rather than GMST. "
            "Months and years are means of the daily series, kept only when every day is present. "
            "The latest days may be preliminary."
        ),
    ),
    Dataset(
        id="JMA",
        name="JMA global average surface temperature",
        producer="Japan Meteorological Agency",
        quantity="GMST",
        native_baseline="1991-2020",
        inputs=("jma_monthly", "jma_annual"),
        parse=parse_jma,
        homepage="https://ds.data.jma.go.jp/tcc/tcc/products/gwp/temp/ann_wld.html",
        licence="Public Data License v1.0 (compatible with CC-BY-4.0)",
        attribution=(
            "Based on JMA global average surface temperature anomalies"
            " (Japan Meteorological Agency),",
            "https://ds.data.jma.go.jp/tcc/tcc/products/gwp/temp/ann_wld.html,",
            "rebased by Climate Resource.",
        ),
        uncertainty="None",
        notes="The latest month may be preliminary.",
    ),
)


def read_native(paths: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every dataset into long monthly and annual frames on its native baseline.

    An annual value is kept only for a year with all twelve monthly values,
    because several producers publish a year-to-date mean for the current year.
    """
    monthly, annual = [], []
    for d in DATASETS:
        m, a = d.parse({name: paths[name] for name in d.inputs})
        m = m.dropna(subset=["value"]).astype({"year": int, "month": int})
        a = a.dropna(subset=["value"]).astype({"year": int})
        months = m[m["statistic"] == "central"].groupby("year").size()
        a = a[a["year"].isin(months[months == len(MONTHS)].index)]
        meta = {"dataset": d.id, "quantity": d.quantity, "baseline": d.native_baseline}
        monthly.append(m.assign(**meta))
        annual.append(a.assign(**meta))
    keys = ["dataset", "quantity", "baseline", "statistic"]
    return (
        pd.concat(monthly)[[*keys, "year", "month", "value"]].reset_index(drop=True),
        pd.concat(annual)[[*keys, "year", "value"]].reset_index(drop=True),
    )


def _period_mean(central: pd.Series, baseline: Baseline) -> float | None:
    window = central.loc[baseline.start : baseline.end].dropna()
    return window.mean() if len(window) == baseline.end - baseline.start + 1 else None


def compute_offsets(
    annual: pd.DataFrame, baselines: list[Baseline], bridge_id: str
) -> pd.DataFrame:
    """Find what to subtract from each native series to put it on each baseline.

    A dataset with every year of a baseline is rebased onto it directly.
    A dataset that starts later is rebased onto the bridge period,
    then shifted by the mean bridge-minus-baseline difference of the datasets that cover both.
    """
    central = annual[annual["statistic"] == "central"].pivot(
        index="year", columns="dataset", values="value"
    )
    means = {b.id: {d: _period_mean(central[d], b) for d in central} for b in baselines}
    bridge_means = means[bridge_id]
    if missing := [d for d, mean in bridge_means.items() if mean is None]:
        raise ValueError(f"{missing} do not cover the bridge period {bridge_id}")

    rows = []
    for b in baselines:
        covering = sorted(d for d, mean in means[b.id].items() if mean is not None)
        gaps = [bridge_means[d] - means[b.id][d] for d in covering]
        gap = sum(gaps) / len(gaps) if gaps else None
        contributors = ", ".join(covering)
        for d in central:
            if means[b.id][d] is not None:
                rows.append((d, b.id, means[b.id][d], "direct", None, None, None))
            elif gap is not None:
                offset = bridge_means[d] - gap
                rows.append((d, b.id, offset, "bridged", bridge_id, gap, contributors))
            else:
                raise ValueError(f"no dataset covers both {b.id} and the bridge {bridge_id}")
    columns = [
        "dataset",
        "baseline",
        "offset",
        "method",
        "bridge",
        "bridge_gap",
        "bridge_datasets",
    ]
    offsets = pd.DataFrame(rows, columns=columns)
    return offsets.round({"offset": DECIMALS, "bridge_gap": DECIMALS})


def rebase(native: pd.DataFrame, offsets: pd.DataFrame) -> pd.DataFrame:
    """Put a native frame on every baseline, shifting the bounds with the central value.

    Monthly values take the annual offset,
    so each keeps the seasonal cycle its producer removed.
    """
    merged = native.drop(columns="baseline").merge(
        offsets[["dataset", "baseline", "offset", "method"]], on="dataset"
    )
    merged["value"] = (merged["value"] - merged["offset"]).round(DECIMALS)
    columns = ["dataset", "quantity", "baseline", "method", "statistic"]
    rest = [c for c in native.columns if c not in columns and c != "value"]
    return merged[[*columns, *rest, "value"]]


def to_timeseries(annual: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long annual frame into the Bookshelf wide timeseries shape."""
    dims = [c for c in annual.columns if c not in ("year", "value")]
    wide = annual.pivot_table(index=dims, columns="year", values="value").reset_index()
    wide.insert(0, "region", "World")
    wide.insert(1, "variable", "Surface Temperature Anomaly")
    wide.insert(2, "unit", "K")
    wide.columns = [str(c) for c in wide.columns]
    return wide


def _table(header: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def render_readme(  # noqa: PLR0913
    *,
    version: str,
    baselines: list[Baseline],
    bridge_id: str,
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    offsets: pd.DataFrame,
    sources: Mapping,
) -> str:
    """Write the README that ships in the book, defining every baseline and offset."""
    retrieved = sources["retrieved"]
    out = [
        "# Global surface temperature",
        "",
        "Observed global mean surface temperature anomalies,",
        f"as their producers published them on {retrieved} (snapshot {version}).",
        "Each series is given on its native baseline and on every baseline below.",
        "The offsets used to move between them are tabled here and published as `offsets`.",
        "",
        "## Entries",
        "",
        _table(
            ["Entry", "Holds"],
            [
                [
                    "`native_annual`",
                    "Annual anomalies on each producer's baseline (timeseries)",
                ],
                ["`native_monthly`", "Monthly anomalies on each producer's baseline"],
                [
                    "`annual`",
                    "Annual anomalies on every baseline, one row set per baseline (timeseries)",
                ],
                ["`monthly`", "Monthly anomalies on every baseline"],
                [
                    "`offsets`",
                    "What was subtracted from each native series for each baseline",
                ],
                ["`readme`", "This document"],
            ],
        ),
        "",
        "Values are in K, which is the same as a difference in °C.",
        "`statistic` is `central`, or `lower` and `upper` where the producer publishes bounds.",
        "`method` is `direct` or `bridged`, as described under [Offsets](#offsets).",
        "",
        "## Datasets",
        "",
    ]
    rows = []
    for d in DATASETS:
        a = annual[(annual["dataset"] == d.id) & (annual["statistic"] == "central")]
        m = monthly[(monthly["dataset"] == d.id) & (monthly["statistic"] == "central")]
        last = m.sort_values(["year", "month"]).iloc[-1]
        rows.append(
            [
                d.id,
                d.producer,
                d.quantity,
                d.native_baseline,
                f"{a['year'].min()}-{a['year'].max()}",
                f"{int(last['year'])}-{int(last['month']):02d}",
                d.uncertainty,
                d.licence,
            ]
        )
    out += [
        _table(
            [
                "Dataset",
                "Producer",
                "Quantity",
                "Native baseline",
                "Complete years",
                "Latest month",
                "Bounds",
                "Licence",
            ],
            rows,
        ),
        "",
        "GMST blends sea surface temperature with air temperature over land and sea ice.",
        "GSAT is air temperature everywhere.",
        "AR6 assesses their long-term changes as identical, differing by at most 10%.",
        "",
    ]
    out += [
        f"- **{d.id}**: {d.name}. {d.notes} <{d.homepage}>".replace(".  ", ". ") for d in DATASETS
    ]
    out += ["", "## Baselines", ""]
    for b in baselines:
        reference = f"[{b.reference}]({b.reference_url})" if b.reference_url else b.reference
        out += [
            f"### {b.id}: {b.name}",
            "",
            b.use.strip(),
            "",
            f"Reference: {reference}.",
            "",
        ]
        for a in b.assessed:
            low, high = a["very_likely"]
            out += [
                f"{a['source']} assesses {a['quantity']} in {b.id} as {a['central']} K "
                f"[{low} to {high}] above {a['since']} (very likely range).",
                "",
            ]

    out += [
        "## Offsets",
        "",
        "A value on a baseline is the native value minus that dataset's offset for the baseline.",
        "Bounds move with the central value,",
        "so they leave out the uncertainty of the baseline mean.",
        "Monthly values take the annual offset,",
        "so each keeps the seasonal cycle its producer removed.",
        "",
        "A dataset with every year of a baseline is rebased onto it directly (`direct`).",
        f"A dataset that starts later is rebased onto {bridge_id},",
        f"then shifted by the mean {bridge_id}-minus-baseline difference",
        "of the datasets that cover both periods (`bridged`).",
        "This is how WMO's State of the Global Climate reaches 1850-1900 for the reanalyses.",
        "",
        "Offsets in K, with bridged offsets marked †:",
        "",
    ]
    ids = [b.id for b in baselines]
    table = offsets.pivot(index="dataset", columns="baseline", values=["offset", "method"])
    grid = [
        [
            d.id,
            *(
                f"{table.loc[d.id, ('offset', i)]:+.3f}"
                + (" †" if table.loc[d.id, ("method", i)] == "bridged" else "")
                for i in ids
            ),
        ]
        for d in DATASETS
    ]
    out += [_table(["Dataset", *ids], grid), ""]
    gaps = offsets[offsets["method"] == "bridged"].drop_duplicates("baseline")
    if len(gaps):
        out += [
            _table(
                ["Baseline", f"Mean {bridge_id} minus baseline (K)", "From"],
                [
                    [g.baseline, f"{g.bridge_gap:+.3f}", g.bridge_datasets]
                    for g in gaps.itertuples()
                ],
            ),
            "",
        ]

    out += [
        "## Sources",
        "",
        _table(
            ["File", "URL", "sha256"],
            [[f"`{n}`", f["url"], f"`{f['sha256'][:12]}`"] for n, f in sources["files"].items()],
        ),
        "",
        "## Licence",
        "",
        "This book is CC-BY-4.0.",
        "Each dataset's own licence, listed above, permits redistributing derived data",
        "with attribution, so credit the producers as well as this book.",
        *(line for d in DATASETS if d.attribution for line in ("", *d.attribution)),
        "",
    ]
    return "\n".join(out)
