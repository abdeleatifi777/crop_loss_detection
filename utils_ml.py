#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hiremas1
"""

import pandas as pd
import numpy as np
import config as cfg
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, MissingIndicator
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.dummy import DummyClassifier

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.4f" % x)
)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

np.random.seed(5)


def load_ndvi_as_df(filepath: str, years):
    df = pd.read_csv(filepath, dtype={2: str, 8: str})
    df.replace(cfg.fill_value, np.nan, inplace=True)
    years = as_list(years)
    df = df.loc[df.year.isin(years)].reset_index(drop=True)
    return df


def load_ndvi_as_numpy(
    filepath: str, years: list, balance_flag: int = 0, synth_data: bool = True
):
    """
    load NDVI univariate time series from csv file
    """
    if synth_data:
        df = pd.read_csv(filepath, index_col=None)
    else:
        df = pd.read_csv(filepath, dtype={2: str, 8: str})
        df = df.loc[df.year.isin(years)].reset_index(drop=True)

    col_names = list(cfg.ts_vars.keys())
    # if balance flag is 1 then sequentially load data per year and balance it
    # this is like stratified sampling across years
    if balance_flag == 1:
        for i, year in enumerate(years):
            df_year = df.loc[df.year == year].reset_index(drop=True)
            x = df_year.loc[:, col_names].values
            y = df_year.loc[:, "loss"].values
            z = df_year.loc[:, "new_ID"].values
            if i == 0:
                x, y, ids = balance_data(x, y, z)
            else:
                x_year, y_year, ids_year = balance_data(x, y, z)
                x = np.vstack([x, x_year])
                y = np.hstack([y, y_year])
                ids = np.hstack([ids, ids_year])
    else:  # bulk load
        x = df.loc[:, col_names].values
        y = df.loss.values
        ids = df["new_ID"].values

    x[x == cfg.fill_value] = np.nan
    x, y, ids = shuffle(x, y, ids)
    x, y = x.astype(np.float32), y.astype(np.int64)

    # # remove columns with all nan values
    # # mask indicating that all the values of a column are 0
    # nan_col_mask = np.all(np.isnan(x), axis=0)
    # x = x.T[~nan_col_mask].T
    return x, y, ids


def balance_data(x, y, ids):
    np.random.seed(cfg.balance_data_seed)
    idx1 = y == 1
    idx0 = y == 0

    x1, y1, ids1 = x[idx1], y[idx1], ids[idx1]
    x0, y0, ids0 = x[idx0], y[idx0], ids[idx0]

    n1 = np.sum(y == 1)
    n0 = np.sum(y == 0)

    idx0 = np.arange(n0)
    idx0 = np.random.permutation(idx0)
    x0, y0, ids0 = x0[idx0[:n1]], y0[idx0[:n1]], ids0[idx0[:n1]]

    x = np.vstack((x0, x1))
    y = np.hstack((y0, y1))
    ids = np.hstack((ids0, ids1))
    return x, y, ids


def shuffle(x, y, ids):
    idx = np.random.permutation(np.arange(len(x)))
    x = x[idx]
    y = y[idx]
    ids = ids[idx]
    return x, y, ids


def compute_average_ndvi():
    filepath = cfg.data_path + cfg.filename
    df = load_ndvi_as_df(filepath, cfg.years)

    ts_vars = list(cfg.ts_vars.keys())
    df = df[["year", "loss"] + ts_vars]
    df = df.groupby(["year", "loss"]).mean()
    df = df.reset_index()
    return df


def create_param_grid(model_name):
    """
    create a parameter grid for the algorithm 'model_name'
    inputs:
        model_name: String with algorithm name
    outputs:
        param_grid: list of all combinations of parameter values
    """
    if model_name == "lr":
        param_grid = {"mdl__penalty": ["l2"], "mdl__C": [1, 10]}

    elif model_name == "dt":
        param_grid = {"mdl__max_depth": [5, 10, 50]}

    elif model_name == "rf":
        param_grid = {"mdl__max_depth": [10], "mdl__n_estimators": [50]}

    elif model_name == "mlp":
        param_grid = {"mdl__hidden_layer_sizes": [(10,), (5,)]}
    return param_grid


def build_model(model_name, imputer_name, random_state):
    if model_name == "lr":
        mdl = LogisticRegression(random_state=random_state)
    elif model_name == "dt":
        mdl = DecisionTreeClassifier(random_state=random_state)
    elif model_name == "rf":
        mdl = RandomForestClassifier(
            random_state=random_state, class_weight="balanced_subsample"
        )
    elif model_name == "mlp":
        mdl = MLPClassifier(random_state=random_state, max_iter=500)
    if imputer_name == "mean":
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    elif imputer_name == "mice":
        imputer = IterativeImputer(
            missing_values=np.nan,
            random_state=random_state,
            n_nearest_features=5,
            sample_posterior=True,
        )
    elif imputer_name == "meanind":
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        imputer = make_union(
            imputer, MissingIndicator(missing_values=np.nan, features="all")
        )
    elif imputer_name == "miceind":
        imputer = IterativeImputer(
            missing_values=np.nan,
            random_state=random_state,
            n_nearest_features=5,
            sample_posterior=True,
        )
        imputer = make_union(
            imputer, MissingIndicator(missing_values=np.nan, features="all")
        )
    scaler = preprocessing.StandardScaler()
    clf = Pipeline([("imputer", imputer), ("scalar", scaler), ("mdl", mdl)])
    return clf


def build_model_opt(model_name, random_state):
    if model_name == "lr":
        mdl = LogisticRegression(random_state=random_state)
    elif model_name == "dt":
        mdl = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    elif model_name == "mlp":
        mdl = MLPClassifier(
            hidden_layer_sizes=(10,), max_iter=500, random_state=random_state,
        )
    elif model_name == "rf":
        mdl = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight="balanced_subsample",
            random_state=random_state,
        )

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer = make_union(
        imputer, MissingIndicator(missing_values=np.nan, features="all")
    )
    scaler = preprocessing.StandardScaler()
    clf = Pipeline([("imputer", imputer), ("scalar", scaler), ("mdl", mdl)])
    return clf


def build_dummy_mdl(random_state):
    mdl = DummyClassifier(strategy="uniform", random_state=random_state)
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer = make_union(
        imputer, MissingIndicator(missing_values=np.nan, features="all")
    )
    scaler = preprocessing.StandardScaler()
    clf = Pipeline([("imputer", imputer), ("scalar", scaler), ("mdl", mdl)])
    return clf


def split_3fold(random_state, x, y, trp=0.6, vap=0.8):
    """
    Split data into train, test and validation set
    """
    assert len(x) == len(y), print("Error: x and y length mismatch.")

    nimages = len(x)
    np.random.seed(random_state)
    idx = np.random.permutation(nimages)
    idx_tr = idx[: int(trp * nimages)]
    idx_va = idx[int(trp * nimages) : int(vap * nimages)]
    idx_te = idx[int(vap * nimages) :]

    X_tr = x[idx_tr]
    y_tr = y[idx_tr]

    X_te = x[idx_te]
    y_te = y[idx_te]

    X_va = x[idx_va]
    y_va = y[idx_va]
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def get500(x, y, ids):
    n = x.shape[0]
    # take 500 random examples of class 1 for test set
    permuted_idx = np.random.permutation(n)
    te_idx = permuted_idx[:500]
    xte = x[te_idx]
    yte = y[te_idx]
    idste = ids[te_idx]

    # remaining are training set
    tr_idx = permuted_idx[500:]
    xtr = x[tr_idx]
    ytr = y[tr_idx]
    idstr = ids[tr_idx]
    return xtr, ytr, idstr, xte, yte, idste


def split_for_map(x, y, ids, seed):
    """
    randomly split data for plotting
    """
    assert len(x) == len(y), print("Error: x and y length mismatch.")

    np.random.seed(seed)

    # split data by class
    x1, y1, ids1 = x[y == 1], y[y == 1], ids[y == 1]
    x0, y0, ids0 = x[y == 0], y[y == 0], ids[y == 0]

    xtr1, ytr1, idstr1, xte1, yte1, idste1 = get500(x1, y1, ids1)
    xtr0, ytr0, idstr0, xte0, yte0, idste0 = get500(x0, y0, ids0)

    xtr = np.vstack([xtr0, xtr1])
    ytr = np.hstack([ytr0, ytr1])
    idstr = np.hstack([idstr0, idstr1])

    xte = np.vstack([xte0, xte1])
    yte = np.hstack([yte0, yte1])
    idste = np.hstack([idste0, idste1])

    return xtr, ytr, idstr, xte, yte, idste


def split_2fold(random_state, x, y, ids, trp=0.8):
    """
    Split data into 2 folds
    """
    assert len(x) == len(y), print("Error: x and y length mismatch.")

    # nimages = len(x)
    # idx = np.random.RandomState(seed=random_state).permutation(nimages)

    # idx_tr = idx[:int(trp*nimages)]
    # idx_te = idx[int(trp*nimages):]

    # X_tr = x[idx_tr]
    # y_tr = y[idx_tr]
    # ids_tr = ids[idx_tr]

    # X_te = x[idx_te]
    # y_te = y[idx_te]
    # ids_te = ids[idx_te]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=trp, random_state=random_state)

    for idx_tr, idx_te in sss.split(x, y):
        X_tr, y_tr, ids_tr = x[idx_tr], y[idx_tr], ids[idx_tr]
        X_te, y_te, ids_te = x[idx_te], y[idx_te], ids[idx_te]

    return X_tr, y_tr, ids_tr, X_te, y_te, ids_te


def match_columns(a, b):
    asum = np.nansum(a, axis=0)
    bsum = np.nansum(b, axis=0)
    keep = (asum != 0) & (bsum != 0)
    aout = a.T[keep]
    bout = b.T[keep]
    return aout.T, bout.T


def as_list(x):
    if type(x) is np.ndarray:
        return x.tolist()
    elif type(x) is list:
        return x
    else:
        return [x]


def permute_like(arr, ref_arr):
    """
    return the indices of elements of arr that are in the same order as
    elements in ref_arr
    """
    index = []
    for val in ref_arr:
        index.append(np.where(arr == val)[0][0])
    return index
