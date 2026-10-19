"""Download today's upstream files into ``inputs/<version>/``.

Upstream files are overwritten in place every month,
so each book reads a checked-in snapshot rather than a live URL.

    uv run python scripts/snapshot.py 2026.10
"""

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

HADCRUT = "https://hadleyserver.metoffice.gov.uk/hadcrut5/data/HadCRUT.5.2.0.0/analysis/diagnostics"
NOAA = "https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/v6.1/access/timeseries"
JMA = "https://ds.data.jma.go.jp/tcc/tcc/products/gwp/temp/list/csv"

FIXED = {
    "hadcrut5_monthly.csv": f"{HADCRUT}/HadCRUT.5.2.0.0.analysis.summary_series.global.monthly.csv",
    "hadcrut5_annual.csv": f"{HADCRUT}/HadCRUT.5.2.0.0.analysis.summary_series.global.annual.csv",
    "gistemp.csv": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
    "era5_daily.csv": "https://sites.ecmwf.int/data/climatepulse/data/series/era5_daily_series_2t_global.csv",
    "jma_monthly.csv": f"{JMA}/mon_wld.csv",
    "jma_annual.csv": f"{JMA}/year_wld.csv",
}


def fetch(url: str) -> bytes:
    """Download ``url``, which is always one of the https sources above."""
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        return response.read()


def noaa_urls() -> dict[str, str]:
    """Find the NOAAGlobalTemp files, whose names carry the month of their latest data."""
    listing = fetch(f"{NOAA}/").decode()
    urls = {}
    for frequency, name in (("mon", "monthly"), ("ann", "annual")):
        pattern = rf'href="(aravg\.{frequency}\.land_ocean\.90S\.90N\.v[\d.]+\.\d{{6}}\.asc)"'
        matches = sorted(set(re.findall(pattern, listing)))
        if not matches:
            raise RuntimeError(f"no NOAAGlobalTemp {name} file in {NOAA}/")
        urls[f"noaaglobaltemp_{name}.asc"] = f"{NOAA}/{matches[-1]}"
    return urls


def main() -> None:
    """Write the snapshot and a ``sources.json`` recording where each file came from."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="the book version, the snapshot month as YYYY.MM")
    version = parser.parse_args().version
    if not re.fullmatch(r"\d{4}\.\d{2}", version):
        parser.error("version must be YYYY.MM")

    target = Path(__file__).parent.parent / "inputs" / version
    target.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name, url in {**FIXED, **noaa_urls()}.items():
        data = fetch(url)
        (target / name).write_bytes(data)
        sources[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
        print(f"{name}: {len(data)} bytes")
    manifest = {
        "retrieved": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": sources,
    }
    (target / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Add a book for version {version} to bookshelf.yaml, copying the previous one.")


if __name__ == "__main__":
    main()
