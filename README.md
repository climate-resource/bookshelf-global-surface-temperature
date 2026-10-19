# Global Surface Temperature

A [Bookshelf](https://github.com/climate-resource/bookshelf) feedstock
for observed global mean surface temperature anomalies from HadCRUT5, NOAAGlobalTemp, GISTEMP, ERA5 and JMA.
Each series is published on its producer's baseline and on the common baselines in `baselines.yaml`.

## What a book holds

- `native_annual` and `native_monthly`: each series on its producer's baseline.
- `annual` and `monthly`: each series on every baseline.
- `offsets`: what was subtracted from each native series for each baseline.
- `readme`: the baseline definitions, offsets, sources and licences.

## Taking a new snapshot

Upstream files are overwritten in place, so each book reads a checked-in monthly snapshot.

```bash
uv run python scripts/snapshot.py 2026.11
```

Add a book for `2026.11` to `bookshelf.yaml`, copying the previous one with the new paths, then build it:

```bash
make run VERSION=2026.11
```

## Adding a baseline

Add an entry to `baselines.yaml`.
The next edition gains rows for it in `annual`, `monthly` and `offsets`.

## Publishing

A pull request builds a preview of each book, and merging publishes it.
See [copier-bookshelf-dataset](https://github.com/climate-resource/copier-bookshelf-dataset) for the rest.
