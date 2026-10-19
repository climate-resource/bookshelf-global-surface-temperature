# Global Surface Temperature

Observed global mean surface temperature anomalies from the major datasets,
published on their native baselines and on a common set of baselines from IPCC AR6 and WMO.

This repository is a feedstock containing code that turns upstream data into books on the
[bookshelf](https://github.com/climate-resource/bookshelf).

## Datasets

| Dataset        | Quantity | Native baseline | Bounds | Source file                     |
| -------------- | -------- | --------------- | ------ | ------------------------------- |
| HadCRUT5       | GMST     | 1961-1990       | 95%    | Summary series, monthly, annual |
| NOAAGlobalTemp | GMST     | 1991-2020       | none   | `aravg` land-ocean 90S-90N      |
| GISTEMP v4     | GMST     | 1951-1980       | none   | `GLB.Ts+dSST.csv`               |
| ERA5           | GSAT     | 1991-2020       | none   | Climate Pulse daily series      |
| JMA            | GMST     | 1991-2020       | none   | Global monthly and annual CSV   |

Berkeley Earth is left out because its CC-BY-NC licence would carry into every derived entry,
and the book is CC-BY-4.0.

JRA-3Q only publishes gridded fields, so it needs its own global averaging before it can join.
DCENT, China-MST and Kadow et al. are research releases on an irregular cadence and are not yet included.

## What a book holds

- `native_annual` and `native_monthly`: each series on its producer's baseline.
- `annual` and `monthly`: each series on every baseline in `baselines.yaml`.
- `offsets`: what was subtracted from each native series for each baseline, and how it was found.
- `readme`: the definition of every baseline, the offsets table and the source of every file.

A dataset with every year of a baseline is rebased onto it directly.
One that starts later, such as ERA5 for 1850-1900,
is rebased onto the bridge period and shifted by the gap the covering datasets agree on.

## Files

- `bookshelf.yaml` is the recipe: the metadata, and one book per monthly snapshot.
- `build.py` reads one snapshot and writes the book.
- `surface_temperature.py` holds the parsers, the rebasing and the README template.
- `baselines.yaml` defines the baselines and the bridge period.
- `scripts/snapshot.py` downloads the upstream files into `inputs/<version>/`.

## Taking a new snapshot

Upstream files are overwritten in place each month, so every book reads a checked-in copy.

```bash
uv run python scripts/snapshot.py 2026.11
```

Then add a book for `2026.11` to `bookshelf.yaml`, copying the previous one with the new paths,
and build it:

```bash
make run VERSION=2026.11
```

## Adding a baseline

Add an entry to `baselines.yaml` with its years, name, use and reference.
The next edition of every book gains rows for it in `annual`, `monthly` and `offsets`,
and a section in the README.
No entry is renamed, so nothing reading the book has to change.

## Publishing

Each pull request builds a preview for each version declared in `bookshelf.yaml`.
Merging publishes that preview to the bookshelf.
Everything else about this feedstock lives in the README of
[copier-bookshelf-dataset](https://github.com/climate-resource/copier-bookshelf-dataset).
