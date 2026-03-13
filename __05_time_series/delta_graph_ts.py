# In[]
#Time series analysis for NTM srikes:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm


from scipy import stats
from scipy.stats import norm

quantile_deltas = pd.read_parquet(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\time_series\quantile_delta.parquet")
