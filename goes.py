# %%
from goes2go import GOES
from goes2go.data import goes_timerange
# %%
start = "2024-07-21 00:00"
end = "2024-07-22 00:00"
g = goes_timerange(start, end, satellite="goes16", product="GLM-L2-LCFA", return_as="filelist")
# %%
