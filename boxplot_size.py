import os
import numpy as np
import pandas as pd
import config as cfg
import matplotlib.pyplot as plt


def make_new(csvdir):

    myall = pd.DataFrame()
    for mycsv in os.listdir(csvdir):
        csvpath = os.path.join(csvdir, mycsv)

        size = mycsv.split('_')[0]
        if size == 'verysmall':
            size = 1
        elif size == 'small':
            size = 2
        elif size == 'medium':
            size = 3
        elif size == 'large':
            size = 4
        df = pd.read_csv(csvpath)
        new_columns = df.columns.values
        new_columns[0] = 'year'
        df.columns = new_columns
        print(df.columns)
        df.insert(0, 'size', size, False)
        myall = pd.concat([myall, df])

    for year in cfg.years:
        print(myall.head())
        yearly = myall[myall['year'] == year]
        yearly = yearly.drop('year', axis=1)
        # collect per year
        plot_boxplot(yearly, cfg.result_base_path +
                     "boxplot_"+str(year)+".png", cfg.save_flag)


def plot_boxplot(df, save_path, save_flag):
    plt.figure(figsize=(25, 10))
    xticks = ['verysmall', 'small', 'medium', 'large']
    xtick_loc = np.arange(len(xticks)) + 1
    print(df.head())
    plt.boxplot(df, showfliers=False, boxprops=dict(linewidth=2.5))
    plt.xticks(xtick_loc, xticks)
    plt.ylim([0, 1])
    plt.xlabel('size')
    plt.ylabel('AUC')
    plt.tight_layout()
    if save_flag:
        plt.savefig(save_path)


csvdir = '/u/58/wittkes3/unix/Documents/aicroppro_spatial/results/auc_size'
make_new(csvdir)
