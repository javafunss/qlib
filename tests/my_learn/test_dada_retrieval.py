# %%
# qlib init
from pandas import Timestamp
import qlib
from qlib.data import D
from qlib.data.filter import NameDFilter
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')

# %%
# 给定的时间范围和频率加载交易日历
cld = D.calendar(start_time='2010-01-01', end_time='2017-12-31', freq='day')[:2]

D.instruments(market='all')

# %%
instruments = D.instruments(market='csi300')
D.list_instruments(instruments=instruments, start_time='2010-01-01', end_time='2017-12-31', as_list=True)[:6]

# %%
nameDFilter = NameDFilter(name_rule_re='SH[0-9]{4}55')
instruments = D.instruments(market='csi300', filter_pipe=[nameDFilter])
D.list_instruments(instruments=instruments, start_time='2015-01-01', end_time='2016-02-15', as_list=True)

# %%
