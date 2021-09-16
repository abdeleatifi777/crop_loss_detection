#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:07:04 2020

@author: hiremas1
"""
import pandas as pd
import numpy as np
import config as cfg
import seaborn as sns
import utils_ml as mut
from scipy.stats import wilcoxon
from scipy.stats import pearsonr
# %% model comparision---------------------------------------------------------


def get_opt_params():
    """
    read hyperparameter optimisation result files to find the optimal parameters
    for each model and imputation strategy 
    """
    columns = ["params",
               "mean_test_score",
               "std_test_score",
               "mean_fit_time",
               "std_fit_time"]
    df_opt = pd.DataFrame(columns=["mdl", "imputer"] + columns)
    for model_name in cfg.model_names:
        for imputer_name in cfg.imputer_names:
            print([model_name, imputer_name])
            filename = f"{model_name}_{imputer_name}_10x10_auc.csv"
            filepath = f"{cfg.result_base_path}optimize/{filename}"

            df = pd.read_csv(filepath)
            temp = df.groupby('params').mean().sort_values(
                'mean_test_score', ascending=False)
            temp.reset_index(inplace=True)
            temp = temp.head(1)
            temp = temp[columns]
            temp["mdl"] = model_name
            temp["imputer"] = imputer_name
            df_opt = pd.concat([df_opt, temp])
    return df_opt


dfopt = get_opt_params()

# %% Within year opt models ------------------------------------------------
# extract average AUC of 10x10 CV


def get_average_auc(df):
    df_new = {}
    for year in cfg.years:
        auc = df[df.year == year].iloc[:, 2:].values.ravel()
        df_new[year] = auc
    df_new = pd.DataFrame(df_new)
    # test for the entire experiment---------------------
    # df with mean and std of AUC values per year
    df_avg_auc = df_new.apply(['mean', 'std']).T
    return df_avg_auc


base_path = 'results/draft/within_year_opt_mdl/unbalanced/'
path_lr = f"{base_path}lr_auc_10x10.csv"
path_dt = f"{base_path}dt_auc_10x10.csv"
path_rf = f"{base_path}rf_auc_10x10.csv"
path_mlp = f"{base_path}mlp_auc_10x10.csv"

dflr = get_average_auc(pd.read_csv(path_lr))
dfdt = get_average_auc(pd.read_csv(path_dt))
dfrf = get_average_auc(pd.read_csv(path_rf))
dfmlp = get_average_auc(pd.read_csv(path_mlp))


# %% within year Wilcoxon test
# wilcoxon of yearly AUC values--------------------
filepath = cfg.result_base_path + cfg.within_years_auc_file
df = pd.read_csv(filepath)
# reformat df to year --> list of auc
df_new = {}
for year in cfg.years:
    auc = df[df.year == year].iloc[:, 2:].values.ravel()
    df_new[year] = auc
df_new = pd.DataFrame(df_new)

# test for the entire experiment---------------------
# df with mean and std of AUC values per year
df_avg_auc = df_new.apply(['mean', 'std']).T

x = df_avg_auc.iloc[:, 0].values

print(wilcoxon(x-0.5, alternative='greater'))

# average AUC per year
#               mean           std
# 2000  0.7568621257  0.0284555323
# 2001  0.6479135041  0.0179438581
# 2002  0.7646845919  0.0229652191
# 2003  0.7056634827  0.0213113763
# 2004  0.6022265139  0.0104871231
# 2005  0.7502768894  0.0308939137
# 2006  0.6567131448  0.0150100704
# 2007  0.6273983269  0.0691652870
# 2008  0.7948579015  0.0083681342
# 2009  0.6473507721  0.0704904538
# 2010  0.6237175915  0.0268402984
# 2011  0.6726666132  0.0235487129
# 2012  0.6363632444  0.0103823428
# 2013  0.6792814431  0.0336165016
# 2014  0.6899675649  0.0293818251
# 2015  0.7549480505  0.0079128055


# %% correlation between average NDVI values
# filepath = cfg.data_path + cfg.filename
# df = ut.load_df(filepath, cfg.years)

# ts_vars = list(cfg.ts_vars.keys())
# df = df[['year', 'loss']+ts_vars]
# df = df.groupby(['year', 'loss']).mean()
# df = df.reset_index()
# df.to_csv('average_ndvi_ts.csv', index=False)
df = pd.read_csv(cfg.result_base_path + 'average_ndvi_ts.csv')
corr = {}
for year in cfg.years:
    a = df[df.year == year]
    a = a.dropna(axis=1)
    c0 = a[a.loss == 0].iloc[0, 2:].values
    c1 = a[a.loss == 1].iloc[0, 2:].values
    corr[year], _ = pearsonr(c0, c1)
df_corr = pd.Series(corr).to_frame().reset_index()
df_corr.columns = ['year', 'ndvi_corr']
# df_corr.to_csv('corr_between_avg_ndvi.csv', index=False)

# -----------------------------------------------------------------------------
# Missing values analysis
# -----------------------------------------------------------------------------


def compute_missing_percent():
    data_path = cfg.data_path+cfg.filename

    arr0 = []
    arr1 = []
    for year in cfg.years:
        print(year)
        X, y, _ = mut.load_ndvi_as_numpy(data_path, mut.as_list(year), 0)
        n0, n1 = np.sum(y == 0), np.sum(y == 1)

        x0 = X[y == 0]
        x1 = X[y == 1]

        mpt0 = np.isnan(x0).sum(axis=0)
        mpt1 = np.isnan(x1).sum(axis=0)
        mpt_year = mpt0 + mpt1

        percent_missing_0 = mpt0/n0
        percent_missing_1 = mpt1/n1
        percent_missing_year = np.sum(mpt_year)/np.prod(X.shape)

        temp0 = [year, n0, n1, percent_missing_year] + list(percent_missing_0)
        temp1 = [year, n0, n1, percent_missing_year] + list(percent_missing_1)

        arr0.append(temp0)
        arr1.append(temp1)

    arr0 = np.array(arr0)
    arr1 = np.array(arr1)

    # first four column
    columns = ['year', 'n0', 'n1', 'percent_missing']
    columns = columns + list(np.arange(X.shape[1]))

    df1 = pd.DataFrame(arr1, columns=columns)
    df1['loss'] = 1

    df0 = pd.DataFrame(arr0, columns=columns)
    df0['loss'] = 0

    df0 = df0.append(df1, ignore_index=True)
    if 0:
        df0.to_csv(cfg.result_base_path + 'missing_percent.csv')

# %% Correlation between the blue and red curves in missing values


def create_mdcorr():
    df = pd.read_csv('results/draft/missing_percent.csv')
    arr0 = df[df.loss == 0, cfg.ts_vars].values
    arr1 = df[df.loss == 1, cfg.ts_vars].values
    corr_miss = []
    for i, year in enumerate(cfg.years):
        corr_miss.append([year, pearsonr(arr0[i, 4:], arr1[i, 4:])[0]])
    df_corr_miss = pd.DataFrame(corr_miss)
    df_corr_miss.columns = ['year', 'pearsonr', 'pvalue']
    if 0:
        df_corr_miss.to_csv('results/draft/missing_data_corr_btw_classes.csv',
                            index=False)

    # # Results [year, r, p-value]
    # [2000, (0.9677125268835121, 2.6176142155646544e-07)]
    # [2001, (0.9974928730193584, 7.768132584205186e-13)]
    # [2002, (0.9712046341072185, 1.4856131654610452e-07)]
    # [2003, (0.9734935313297016, 9.856584909331123e-08)]
    # [2004, (0.9474343394932961, 2.892860429373375e-06)]
    # [2005, (0.7717499502949063, 0.0032774234814097734)]
    # [2006, (0.9186972085441042, 2.4378163933944347e-05)]
    # [2007, (0.8738535759421964, 0.0002027967907501111)]
    # [2008, (0.8628482751832142, 0.00030221036070153713)]
    # [2009, (0.9647937414696165, 4.0150833739487914e-07)]
    # [2010, (0.984134383908064, 7.70932805146738e-09)]
    # [2011, (0.9911811707957033, 4.1391502915760365e-10)]
    # [2012, (0.9833664089612882, 9.752199550892417e-09)]
    # [2013, (0.9868317386293626, 3.0502628857400265e-09)]
    # [2014, (0.9519124569160509, 1.8676021050968515e-06)]
    # [2015, (0.9369342222747734, 7.0634152931548125e-06)]


# corr_miss = [0.9677125268835121,
# 0.9974928730193584,
# 0.9712046341072185,
# 0.9734935313297016,
# 0.9474343394932961,
# 0.7717499502949063,
# 0.9186972085441042,
# 0.8738535759421964,
# 0.8628482751832142,
# 0.9647937414696165,
# 0.984134383908064,
# 0.9911811707957033,
# 0.9833664089612882,
# 0.9868317386293626,
# 0.9519124569160509,
# 0.9369342222747734]


# years = np.arange(2000, 2016)
# auc = [0.7568, 0.6479, 0.7647, 0.7057, 0.6022, 0.7503, 0.6567, 0.6274,
#        0.7949, 0.6474, 0.6237, 0.6727, 0.6364, 0.6792,  0.6900,  0.7549]
