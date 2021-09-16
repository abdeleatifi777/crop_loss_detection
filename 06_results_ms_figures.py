#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:32:49 2020

@author: hiremas1
"""

import numpy as np
import pandas as pd
import config as cfg
import seaborn as sns
import utils_ls as lut
import utils_ml as mut
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
fontsize = 25
plt.rcParams.update({'font.size': fontsize})
plt.rcParams['xtick.major.pad'] = '8'
plt.rcParams['ytick.major.pad'] = '8'

sns.set(context='notebook',
        style='white',
        palette='deep',
        font='sans-serif',
        font_scale=1.5,
        color_codes=False,
        rc=None)
# -------------------------------------------------------------------------------


# %%
def plot_avg_ndvi(save_figure=0):
    filepath = cfg.data_path + cfg.filename
    df = mut.load_ndvi_as_df(filepath, cfg.years)

    ts_vars = list(cfg.ts_vars.keys())
    df = df[['year', 'loss']+ts_vars]
    df = df.groupby(['year', 'loss']).mean()
    df = df.reset_index()

    # axes objects
    x = np.arange(len(df.columns)-2)
    xticklabels = cfg.xticklabels

    # correlation between the average red and blue curves for each year
    corr = np.array([[0.9528, 0.9971, 0.9306, 0.9220],
                     [0.9959, 0.9593, 0.9941, 0.9735],
                     [0.9644, 0.9507, 0.9945, 0.9833],
                     [0.9878, 0.9921, 0.9796, 0.9746]])
    corr = np.round(corr, 2)
    # outer axes for common x and y label
    fig, axes = plt.subplots(nrows=4, ncols=4)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none',
                    top=False,
                    bottom=False,
                    right=False,
                    pad=25)
    # plt.xlabel("common X")
    plt.ylabel("Average NDVI", fontsize=25)
    plt.xlabel("Time points", fontsize=25)

    years = np.arange(2000, 2016).reshape((4, 4))
    for i in np.arange(4):
        for j in np.arange(4):
            year = years[i, j]
            ndvi = df.loc[df.year == year, df.columns[1:]]
            ndvi1 = ndvi.loc[ndvi.loss == 1, ndvi.columns[1:]].values[0]
            ndvi0 = ndvi.loc[ndvi.loss == 0, ndvi.columns[1:]].values[0]
            print(corr[i, j])
            text = f'{corr[i, j]}'
            axes[i, j].plot(x, ndvi0, marker='o', color='b')
            axes[i, j].plot(x, ndvi1, marker='o', color='r')
            axes[i, j].set_ylim([-0.1, 1])
            axes[i, j].set_title(year)
            axes[i, j].annotate(text, xy=(x[1], 0.7))
            if i != 3:
                axes[i, j].set_xticks([])
                axes[i, j].set_xticklabels([])
            else:
                axes[i, j].set_xticks(x)
                axes[i, j].set_xticklabels(xticklabels)
            if j != 0:
                axes[i, j].set_yticks([])
                axes[i, j].set_yticklabels([])
    axes[0, 3].legend(['noloss', 'loss'], loc="lower right")
    plt.show()
    if save_figure:
        plt.savefig('avg_ndvi.png')

# %%


def plot_ndvi_uts_examples(save_figure=0, nexamples=20):
    filepath = cfg.data_path + cfg.filename
    df = mut.load_ndvi_as_df(filepath, cfg.years)
    ts_vars = list(cfg.ts_vars.keys())
    df = df[['year', 'loss']+ts_vars]

    x = np.arange(len(df.columns)-2)
    xticklabels = cfg.xticklabels

    fig, axes = plt.subplots(nrows=4, ncols=4)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none',
                    top=False,
                    bottom=False,
                    right=False)
    # plt.xlabel("common X")
    plt.ylabel("NDVI", fontsize=25)
    plt.xlabel("Time points", fontsize=25)

    years = cfg.years.reshape((4, 4))
    for i in np.arange(4):
        for j in np.arange(4):
            year = years[i, j]
            ndvi = df.loc[df.year == year, df.columns[1:]]
            n1 = np.sum(ndvi.loss == 1)
            n0 = np.sum(ndvi.loss == 0)

            idx1 = np.random.permutation(np.arange(n1))
            idx0 = np.random.permutation(np.arange(n0))

            idx1 = idx1[:nexamples]
            idx0 = idx0[:nexamples]

            ndvi1 = ndvi.loc[ndvi.loss == 1, ndvi.columns[1:]].values[idx1]
            ndvi0 = ndvi.loc[ndvi.loss == 0, ndvi.columns[1:]].values[idx0]
            axes[i, j].plot(ndvi0.T, marker='o', color='b', label='noloss')
            axes[i, j].plot(ndvi1.T, marker='o', color='r', label='loss')
            axes[i, j].set_ylim([-0.5, 1.1])
            axes[i, j].set_title(year)
            if i != 3:
                axes[i, j].set_xticks([])
                axes[i, j].set_xticklabels([])
            else:
                axes[i, j].set_xticks(x)
                axes[i, j].set_xticklabels(xticklabels)
            if j != 0:
                axes[i, j].set_yticks([])
                axes[i, j].set_yticklabels([])
    if save_figure:
        plt.savefig('ndvi_examples.png')

# %% missing value percentage graphs in different plots


def plot_missing_profile(save_figure=0):
    df = pd.read_csv(cfg.result_base_path + 'missing_percent.csv')
    df1 = df[df['loss'] == 1]
    df1.drop(columns='loss', inplace=True)
    df0 = df[df['loss'] == 0]
    df0.drop(columns='loss', inplace=True)

    x = np.arange(len(df0.columns)-4)
    xticklabels = cfg.xticklabels

    fig, axes = plt.subplots(nrows=4, ncols=4)

    # outer axes for common x and y label
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none',
                    top=False,
                    bottom=False,
                    right=False,
                    pad=25)
    # plt.xlabel("common X")
    plt.ylabel("% data available", fontsize=25)
    plt.xlabel("Time points", fontsize=25)

    years = cfg.years.reshape((4, 4))
    for i in np.arange(4):
        for j in np.arange(4):
            year = years[i, j]
            # row = df0.loc[df0.year == year]
            # row1 = df1.loc[df1.year == year]
            # n0, n1 = int(row.n0), int(row.n1)
            # text = f'n1:{n1}\nn0={n0}\nan={yearly_missing}'
            missing_t = df0.loc[df0.year == year, df0.columns[4:]].values[0]
            missing_t1 = df1.loc[df1.year == year, df1.columns[4:]].values[0]
            # corr, _ = np.round(pearsonr(missing_t, missing_t1), 2)
            # text = f'{corr}'
            axes[i, j].plot(x, missing_t, marker='o', color='b')
            axes[i, j].plot(x, missing_t1, marker='o', color='r')
            axes[i, j].set_ylim([0, 1.2])
            axes[i, j].set_title(year)
            # axes[i, j].annotate(text, xy=(x[0], 0.1))
            if i != 3:
                axes[i, j].set_xticks([])
                axes[i, j].set_xticklabels([])
            else:
                axes[i, j].set_xticks(x)
                axes[i, j].set_xticklabels(xticklabels)
            if j != 0:
                axes[i, j].set_yticks([])
                axes[i, j].set_yticklabels([])
    axes[0, 3].legend(['noloss', 'loss'], loc="lower right")
    plt.show()


# %%


def scatter_auc_vs_missing_percent():
    # missing percentage
    filepath = cfg.result_base_path + 'missing_percent.csv'
    df = pd.read_csv(filepath)

    df1 = df[df['loss'] == 1]
    # df0 = df[df['loss']==0]
    y = df1['percent_missing'].values

    # average within AUC
    dfauc = pd.read_csv(cfg.result_base_path +
                        'within_year_opt_mdl/unbalanced/auc_10x10_mean_std.csv')
    x = dfauc['auc_mean'].values
    plt.figure(figsize=(10, 7))
    ax = sns.regplot(x, y, ci=0, scatter_kws={'s': 100})
    # ax.set_ylim([None, 1])
    ax.set_xlabel('Within-year AUC', fontsize=fontsize)
    ax.set_ylabel('% missing data', fontsize=fontsize)

# %%


def scatter_auc_vs_missing_corr():
    # corr between red and blue curve of missing % plot
    filepath = cfg.result_base_path + 'missing_data_corr_btw_classes.csv'
    df = pd.read_csv(filepath)
    y = df['corr'].values

    # average within AUC
    dfauc = pd.read_csv(cfg.result_base_path +
                        'within_year_opt_mdl/unbalanced/auc_10x10_mean_std.csv')
    x = dfauc['auc_mean']
    plt.figure(figsize=(10, 7))
    ax = sns.regplot(x, y, ci=0, scatter_kws={'s': 100})
    ax.set_ylim([None, 1])
    ax.set_xlabel('Within-year AUC', fontsize=fontsize)
    ax.set_ylabel('MC-corr', fontsize=fontsize)

# %% scatter plot between-year experiment
#  average AUC and correlation between average NDVI


def plot_scatter_auc_vs_ndvi_corr():
    import seaborn as sns
    dfcorr = pd.read_csv(cfg.result_base_path + 'corr_between_avg_ndvi.csv')
    y = dfcorr['ndvi_corr'].values

    dfauc = pd.read_csv(cfg.result_base_path +
                        'within_year_opt_mdl/unbalanced/auc_10x10_mean_std.csv')
    x = dfauc['auc_mean'].values
    plt.figure(figsize=(10, 7))
    ax = sns.regplot(x, y, ci=0, scatter_kws={'s': 100})
    ax.set_ylim([None, 1])
    ax.set_ylabel('NDVI-corr', fontsize=fontsize)
    ax.set_xlabel('Within-year AUC', fontsize=fontsize)

# %% Between years single-year training heatmap
# heatmatp plot


def plot_single_year_heatmatp():
    import matplotlib.pyplot as plt
    import seaborn as sns
    # sns.set(font_scale = 1.5)
    filepath = (cfg.result_base_path +
                "btw_years/unbalanced/auc_rf_single.csv")
    df = pd.read_csv(filepath, index_col='test_year')
    a = df.values
    np.fill_diagonal(a, np.nan)
    yticks = df.index
    xticks = df.columns
    ax = sns.heatmap(df, annot=True,
                     fmt=".4f",
                     linewidths=.5,
                     xticklabels=xticks,
                     yticklabels=yticks)
    plt.xticks(rotation=0)
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0)
    plt.ylabel('Test years', fontsize=15)
    plt.xlabel('Train years', fontsize=15)
    plt.show()


def plot_single_year_half_heatmatp():
    import matplotlib.pyplot as plt
    import seaborn as sns
    # sns.set(font_scale = 1.5)
    filepath = cfg.result_base_path + "btw_years/unbalanced/auc_rf_single.csv"
    df = pd.read_csv(filepath, index_col='test_year')
    a = df.values
    np.fill_diagonal(a, np.nan)
    b = a.T
    c = (a+b)/2
    mask = np.zeros_like(c)
    mask[np.tril_indices_from(mask)] = True
    yticks = df.index
    xticks = df.columns
    with sns.axes_style("white"):
        ax = sns.heatmap(c,
                         annot=True,
                         mask=mask,
                         fmt=".4f",
                         linewidths=.5,
                         square=True,
                         xticklabels=xticks,
                         yticklabels=yticks)
    plt.xticks(rotation=0)
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0)
    plt.ylabel('Test years', fontsize=15)
    plt.xlabel('Train years', fontsize=15)
    plt.show()

# %% within year boxplot


def plot_within_year_auc_boxplot():
    # Boxplots
    # within year  experiment---------------------------------
    import seaborn as sns
    expname = "within_year_opt_mdl/unbalanced/"
    filename = cfg.result_base_path + expname + "auc.csv"

    df = pd.read_csv(filename, index_col=0)
    df = df.iloc[:, :-5]

    ax = sns.boxplot(data=df.T, color="darkturquoise", width=0.7, linewidth=3)
    ax.set_ylabel("AUC", fontsize=50)
    ax.set_xlabel("Year", fontsize=50)
    ax.set_ylim([0.4, 1])

    # Between year single year-----------------------------------
    expname = "btw_years/unbalanced/"
    df = pd.read_csv(filename, index_col=0)
    # ax = df.boxplot(grid=False,
    #                 fontsize=35,
    #                 rot=45,
    #                 widths=0.7,
    #                 boxprops=dict(linewidth=3))

    ax = sns.boxplot(data=df, color="darkturquoise", width=0.7, linewidth=3)
    ax.set_xlabel("Year", fontsize=50)
    ax.set_ylabel("AUC", fontsize=50)
    ax.set_ylim([0.4, 1])


# %%
def imshow_band_seq(im_seq, results_path, expname, save_figure=0):
    """
    plot the time series of each spectral band and also NDVI and RGB
    INPUTS:
        1) im_seq: numpy array of image sequence
        2) results_path: path where the plot is to be saved.
        3) save_figure: flag indicating whether or not to save the plot
    """
    title = "image sequence",
    seq_len, nbands, _, _ = im_seq.shape
    nrows = nbands + 2
    ncols = seq_len
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
    fig.suptitle(title, fontsize=30)
    for t in np.arange(ncols):
        imt = im_seq[t]
        rgbt = lut.compute_rgb(imt)
        ndvit = lut.compute_ndvi(imt)
        for i in np.arange(nbands):
            axarr[i, t].axis("off")
            axarr[i, t].imshow(imt[i])

        axarr[-2, t].axis("off")
        axarr[-2, t].imshow(ndvit)

        axarr[-1, t].axis("off")
        axarr[-1, t].imshow(rgbt)

    plt.subplots_adjust(wspace=0, hspace=0.05, left=0,
                        right=1, bottom=0.01, top=0.90)

#    plt.tight_layout()
    if save_figure == 1:
        plt.savefig(expname+'_im_seq.png')
        plt.close(fig)
    return


def imshow_ndvi_rgb_seq(im_seq, results_path, expname, save_figure=0):
    """
    plot the time series of each spectral band and also NDVI and RGB
    INPUTS:
        1) im_seq: numpy array of image sequence
        2) results_path: path where the plot is to be saved.
        3) save_figure: flag indicating whether or not to save the plot
    """
    title = "image sequence",
    seq_len, _, _, _ = im_seq.shape
    ncols = seq_len
    fig, axarr = plt.subplots(nrows=2, ncols=ncols, figsize=(20, 20))
    fig.suptitle(title, fontsize=30)
    for t in np.arange(ncols):
        imt = im_seq[t]
        rgbt = lut.compute_rgb(imt)
        ndvit = lut.compute_ndvi(imt)

        axarr[-2, t].axis("off")
        axarr[-2, t].imshow(ndvit)

        axarr[-1, t].axis("off")
        axarr[-1, t].imshow(rgbt)

    plt.subplots_adjust(wspace=0, hspace=0.05, left=0,
                        right=1, bottom=0.01, top=0.90)

#    plt.tight_layout()
    if save_figure == 1:
        plt.savefig(expname+'_im_seq.png')
        plt.close(fig)
    return

# %%


def plot_learning_curve(estimator, X, Y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Credit: https://tinyurl.com/yb89spa6

    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    Y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``Y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``Y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    from sklearn.model_selection import learning_curve
    plt.figure()
    plt.title('Learning Curve')
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, Y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
