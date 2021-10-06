"""
Within year classification
"""
# %%
import os
import numpy as np
import pandas as pd
import config as cfg
import utils_ml as mut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

expname = "within_year_opt_mdl"
data_path = cfg.data_path + cfg.filename
result_path = cfg.result_base_path + expname

print(f"experiment: {expname}")
for opt_mdl in cfg.model_names:
    columns = ["auc_cv_" + str(i) for i in np.arange(1, 11)]
    columns = ["year", "seed"] + columns
    df_result = pd.DataFrame(columns=columns)
    print()
    print(f"model: {opt_mdl}")
    for year in cfg.years:
        print(f"year: {year}")
        x, y, ids = mut.load_ndvi_as_numpy(
            data_path, mut.as_list(year), balance_flag=0, synth_data=True
        )
        for i, random_state in enumerate(cfg.random_states):
            print(f"seed: {random_state}", end=", ")
            sss = StratifiedShuffleSplit(
                n_splits=cfg.cv, train_size=cfg.train_size, random_state=random_state
            )
            clf = mut.build_model_opt(opt_mdl, random_state)
            auc_cv = cross_val_score(clf, x, y, cv=sss, scoring="roc_auc")

            # store auc values in a table
            dfi = pd.Series([year, random_state] + list(auc_cv), index=columns)
            df_result = df_result.append(dfi, ignore_index=True)
            print(f"auc: {np.mean(auc_cv):5f}+-{np.std(auc_cv):5f}")
    if 0:
        destination_file = f"{result_path}{opt_mdl}_auc_10x10.csv"
        df_result.to_csv(destination_file, index=False)
