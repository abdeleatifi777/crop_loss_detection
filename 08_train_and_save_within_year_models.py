"""
Train and save the optimal (RF+Mean+MI) within-year models for each year
"""
# %%
import config as cfg
import utils_ml as mut
import joblib

data_path = cfg.data_path + cfg.ml_ready_ndvi_data
opt_mdl = cfg.best_model
for year in cfg.years:
    x, y, ids = mut.load_ndvi_as_numpy(data_path, mut.as_list(year), balance_flag=0)
    print(f"year: {year}, shape:{x.shape}")
    clf = mut.build_model_opt(opt_mdl, random_state=99)
    clf = clf.fit(x, y)
    _ = joblib.dump(clf, f"within_year_rf_mean_mi_{year}.pkl")
