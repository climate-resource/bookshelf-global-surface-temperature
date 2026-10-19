# %% [markdown]
# # Global surface temperature
#
# Reads one snapshot of the upstream global series,
# then publishes them on their native baselines, on every baseline in `baselines.yaml`,
# the offsets between the two, and a README defining both.

# %%
import json
from pathlib import Path

import bookshelf

from surface_temperature import (
    BOOK_DATASETS,
    DATASETS,
    compare_assessments,
    compute_offsets,
    load_assessments,
    load_baselines,
    read_native,
    rebase,
    render_readme,
    to_timeseries,
)

# %%
build = bookshelf.setup()
version = build.book.metadata.version
datasets = BOOK_DATASETS.get(version, DATASETS)
baselines, bridge = load_baselines(Path("baselines.yaml"))
assessments = load_assessments(Path("assessments.yaml"), baselines)

# %%
names = list(dict.fromkeys(name for d in datasets for name in d.inputs))
inputs = {name: build.use(name) for name in [*names, "sources"]}
native_monthly, native_annual = read_native({name: inputs[name].path for name in names}, datasets)
native_annual.groupby(["dataset", "statistic"])["year"].agg(["min", "max", "size"])

# %% [markdown]
# ## Rebase

# %%
offsets = compute_offsets(native_annual, baselines, bridge)
annual = rebase(native_annual, offsets)
monthly = rebase(native_monthly, offsets)
offsets.pivot(index="dataset", columns="baseline", values="offset")

# %% [markdown]
# Each dataset beside the published estimates in `assessments.yaml`.

# %%
comparison = compare_assessments(annual, assessments)
comparison[["reference", "period", "assessed", "dataset", "observed", "difference"]]

# %% [markdown]
# ## Record

# %%
used = list(inputs.values())
build.book.write("native_annual", to_timeseries(native_annual), type="timeseries", used=used)
build.book.write("native_monthly", native_monthly, type="tabular", used=used)
build.book.write("annual", to_timeseries(annual), type="timeseries", used=used)
build.book.write("monthly", monthly, type="tabular", used=used)
build.book.write("offsets", offsets, type="tabular", used=used)
build.book.write("assessments", comparison, type="tabular", used=used)

readme = render_readme(
    version=version,
    datasets=datasets,
    baselines=baselines,
    bridge_id=bridge,
    monthly=native_monthly,
    annual=native_annual,
    offsets=offsets,
    comparison=comparison,
    sources=json.loads(Path(inputs["sources"].path).read_text()),
)
build.book.write("readme", readme.encode(), type="document", used=used, format="md")
build.book.publish()
