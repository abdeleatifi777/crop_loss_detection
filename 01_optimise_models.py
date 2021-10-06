"""
Optimize classifiers using data from year 2015
"""
import os
import pandas as pd
import config as cfg
import utils_ml as mut
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

result_path = cfg.result_base_path + "optimise/"

if not os.path.exists(result_path):
    os.makedirs(result_path)

data_path = cfg.data_path + cfg.filename
x, y, _ = mut.load_ndvi_as_numpy(
    data_path, mut.as_list(2015), balance_flag=1, synth_data=True
)

for model_name in cfg.model_names:
    for imputer_name in cfg.imputer_names:
        print()
        print(f"Model {[model_name, imputer_name]}")
        df = pd.DataFrame([])
        for i, random_state in enumerate(cfg.random_states):
            print(f"seed: {random_state}")
            sss = StratifiedShuffleSplit(
                n_splits=cfg.cv, train_size=cfg.train_size, random_state=random_state
            )

            clf = mut.build_model(model_name, imputer_name, random_state)
            param_grid = mut.create_param_grid(model_name)
            grid_results = GridSearchCV(
                clf, param_grid=param_grid, scoring="roc_auc", cv=sss, verbose=0
            )
            grid_results.fit(x, y)
            dfi = pd.DataFrame(grid_results.cv_results_)
            dfi["seed"] = random_state
            if i == 0:
                df = dfi
            else:
                df = df.append(dfi, ignore_index=True)
        print(f"10x10 auc df shape: {df.shape}")
        if 0:
            csvname = f"{model_name}_{imputer_name}_10x10_auc.csv"
            df.to_csv(result_path + csvname, index=False)
