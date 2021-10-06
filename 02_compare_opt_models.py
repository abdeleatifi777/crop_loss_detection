"""
Compare optimum models for all years
"""
import numpy as np
import config as cfg
import utils_ml as mut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

data_path = cfg.data_path + cfg.filename
sss = StratifiedShuffleSplit(
    n_splits=cfg.cv, train_size=cfg.train_size, random_state=99
)
for year in [2015]:
    print(year)
    X, y, ids = mut.load_ndvi_as_numpy(
        data_path, mut.as_list(year), balance_flag=0, synth_data=True
    )
    for model_name in ["lr", "dt", "mlp", "rf"]:
        clf = mut.build_model_opt(model_name, 99)
        auc_cv = cross_val_score(clf, X, y, cv=sss, scoring="roc_auc")
        print(f"{model_name}, auc={np.mean(auc_cv):5f}+-{np.std(auc_cv):5f}")
