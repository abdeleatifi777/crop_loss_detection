"""
RF_all
train years: not k (all years except kth year)
test year: k
"""
import numpy as np
import pandas as pd
import config as cfg
import utils_ml as mut
from sklearn.metrics import roc_auc_score

expname = "btw_years"
data_path = cfg.data_path + cfg.filename
result_path = cfg.result_base_path + expname + "/unbalanced/"

clf = mut.build_model_opt(cfg.best_model, cfg.random_state_btw)
print(expname)
df_result = pd.DataFrame(index=cfg.years, columns=["aucte"])
df_result.index.name = "test_years"

years = cfg.years
years = np.delete(years, -1)
for yearte in years:
    yeartr = list(set(list(years)) - set([yearte]))

    print(f"test year: {yearte}", end="--->")

    xtr, ytr, _ = mut.load_ndvi_as_numpy(
        data_path, mut.as_list(yeartr), balance_flag=0, synth_data=True
    )

    xte, yte, _ = mut.load_ndvi_as_numpy(
        data_path, mut.as_list(yearte), balance_flag=0, synth_data=True
    )

    xtr, xte = mut.match_columns(xtr, xte)

    clf.fit(xtr, ytr)
    yhatte_proba = clf.predict_proba(xte)
    aucte = roc_auc_score(yte, yhatte_proba[:, 1])
    print(f"aucte:{aucte:5f}")
    df_result.loc[yearte, "aucte"] = aucte

if 0:
    df_result.to_csv(result_path + "auc_rf_all.csv")
print(df_result)
