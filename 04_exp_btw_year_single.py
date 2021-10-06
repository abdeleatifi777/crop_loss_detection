"""
RF_single
train years: k
test year: any year other than k
"""
import os
import numpy as np
import pandas as pd
import utils_ml as mut
import config as cfg
from sklearn.metrics import roc_auc_score

expname = "btw_years"
data_path = cfg.data_path + cfg.filename
result_path = cfg.result_base_path + expname
if not os.path.exists:
    os.makedirs(result_path)


# dummy model or the optimal model
dummy_clf_flag = 0
if dummy_clf_flag:
    clf = mut.build_dummy_mdl(cfg.random_state_btw)
    dst_file = "auc_rf_single_dummy_uniform.csv"
else:
    clf = mut.build_model_opt(cfg.best_model, cfg.random_state_btw)
    dst_file = "auc_rf_single.csv"

print(expname)
df_result = pd.DataFrame(index=cfg.years, columns=cfg.years)
df_result.index.name = "test_years"

for yeartr in cfg.years:
    print(f"train year: {yeartr}")
    for yearte in cfg.years:
        print(f"test year: {yearte}", end="-->")

        xtr, ytr, _ = mut.load_ndvi_as_numpy(
            data_path, mut.as_list(yeartr), balance_flag=0, synth_data=True,
        )

        xte, yte, _ = mut.load_ndvi_as_numpy(
            data_path, mut.as_list(yearte), balance_flag=0, synth_data=True,
        )

        xtr, xte = mut.match_columns(xtr, xte)

        clf.fit(xtr, ytr)

        yhatte_proba = clf.predict_proba(xte)
        auc = roc_auc_score(yte, yhatte_proba[:, 1])
        print(f"auc: {auc:5f}")
        df_result.loc[yearte, yeartr] = auc
if 0:
    df_result.to_csv(result_path + dst_file)
print(df_result)
