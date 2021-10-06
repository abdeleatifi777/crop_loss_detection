# Santoshs paths and possible other configurations
import numpy as np
import pandas as pd

pd.set_option("precision", 10)  # display precision
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 300)

# filename = "ndvi_1_30_365_all_years.csv"
# ml_ready_ndvi_data = "ndvi_1_30_365_all_years_no_ndvieq1.csv"
filename = "synthetic_ndvi_ts.csv"

data_path = "data/"
result_base_path = "results/"
attrib_path = "data/landsat/qa/"
image_path = "data/landsat/qa/"
optimise_exp_path = result_base_path + "optimise/"
within_year_exp_path = result_base_path + "within_year_opt_mdl/unbalanced/"

within_years_auc_file = "within_year_opt_mdl/unbalanced/auc_10x10.csv"
btw_years_single_auc_file = "btw_years/unbalanced/auc_rf_single.csv"
btw_years_all_auc_file = "btw_years/unbalanced/auc_rf_all.csv"

avg_ndvi_file = result_base_path + "average_ndvi_ts.csv"

save_flag = 1
balance_flag = 0
nclass = 2
random_states = [99, 78, 61, 16, 73, 8, 62, 27, 30, 80]
random_state_btw = 42
balance_data_seed = 42
cv = 10
test_size = 0.2
train_size = 1 - test_size
fill_value = -9999
decimal_precision = 5
years = np.arange(2000, 2016)

attribute_vars = [
    "new_ID",
    "year",
    "orig_ID",
    "full_cl",
    "partial_cl",
    "loss",
    "area",
    "plantcode",
    "speciescod",
    "farmid",
]

# time stamp names to doy range dictionary
ts_vars = {
    "t1": "1-30",
    "t2": "31-60",
    "t3": "61-90",
    "t4": "91-120",
    "t5": "121-150",
    "t6": "151-180",
    "t7": "181-210",
    "t8": "211-240",
    "t9": "241-270",
    "t10": "271-300",
    "t11": "301-330",
    "t12": "331-360",
}

xticklabels = [
    "$t_1$",
    "$t_2$",
    "$t_3$",
    "$t_4$",
    "$t_5$",
    "$t_6$",
    "$t_7$",
    "$t_8$",
    "$t_9$",
    "$t_{10}$",
    "$t_{11}$",
    "$t_{12}$",
]


model_names = ["rf", "lr", "dt", "mlp"]
imputer_names = ["mean", "mice", "meanind", "miceind"]
best_model = "rf"
