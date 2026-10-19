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
    DATASETS,
    compute_offsets,
    load_baselines,
    read_native,
    rebase,
    render_readme,
    to_timeseries,
)

# %%
build = bookshelf.setup()
baselines, bridge = load_baselines(Path("baselines.yaml"))

# %%
names = [name for d in DATASETS for name in d.inputs]
inputs = {name: build.use(name) for name in [*names, "sources"]}
native_monthly, native_annual = read_native({name: inputs[name].path for name in names})
native_annual.groupby(["dataset", "statistic"])["year"].agg(["min", "max", "size"])

# %% [markdown]
# ## Rebase

# %%
offsets = compute_offsets(native_annual, baselines, bridge)
annual = rebase(native_annual, offsets)
monthly = rebase(native_monthly, offsets)
offsets.pivot(index="dataset", columns="baseline", values="offset")

# %% [markdown]
# Every dataset on 1850-1900, as a sense check against AR6's 1.09 K for 2011-2020.

# %%
central = annual[(annual["baseline"] == "1850-1900") & (annual["statistic"] == "central")]
central[central["year"].between(2011, 2020)].groupby("dataset")["value"].mean().round(2)

# %% [markdown]
# ## Record

# %%
used = list(inputs.values())
build.book.write("native_annual", to_timeseries(native_annual), type="timeseries", used=used)
build.book.write("native_monthly", native_monthly, type="tabular", used=used)
build.book.write("annual", to_timeseries(annual), type="timeseries", used=used)
build.book.write("monthly", monthly, type="tabular", used=used)
build.book.write("offsets", offsets, type="tabular", used=used)

readme = render_readme(
    version=build.book.metadata.version,
    baselines=baselines,
    bridge_id=bridge,
    monthly=native_monthly,
    annual=native_annual,
    offsets=offsets,
    sources=json.loads(Path(inputs["sources"].path).read_text()),
)
build.book.write("readme", readme.encode(), type="document", used=used, format="md")
build.book.publish()
