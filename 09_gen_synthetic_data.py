"""
Generate synthetic data for verifying that code runs
"""
import numpy as np
import pandas as pd
import config as cfg
from scipy.stats import beta
import matplotlib.pyplot as plt


def pixel_seq(label, seq_len=12):
    """
    Generate single time series from beta pdf 
    Use beta pdf because it can yield a unimodal curve of any shape within [0,1]
    """
    x = np.linspace(0, 1, seq_len)
    if label == "loss":
        # beta distribution parameters for loss
        a, b, scale = 4, 5, 0.1
    else:
        # beta distribution parameters for no loss
        a, b, scale = 5, 4.5, 0.08

    uts = beta.pdf(x, a, b, scale)

    # add noise
    factor = np.random.choice([3.4, 3.6])
    sigma = np.var(uts) / factor
    noise = np.random.normal(uts, sigma, seq_len)
    uts += noise
    uts = (uts - min(uts)) / (max(uts) - min(uts))
    return uts


if __name__ == "__main__":
    N = 2000
    print(f"generating {N} time series for each year")
    n0 = int(N / 2)
    columns = ["new_ID", "loss", "year"] + list(cfg.ts_vars.keys())
    df = pd.DataFrame(columns=columns)

    for year in cfg.years:
        seq_len = 12
        x0 = np.zeros((n0, seq_len))
        x1 = np.zeros((n0, seq_len))
        for i in np.arange(n0):
            x0[i, :] = pixel_seq("noloss")
            x1[i, :] = pixel_seq("loss")

        # plt.plot(x0[0])
        # plt.plot(x1[1])
        # plt.legend(["no-loss", "loss"])
        # plt.show()

        ts = np.vstack((x0, x1))
        loss = np.vstack((np.zeros((n0, 1)), np.ones((n0, 1))))
        new_ID = np.arange(len(ts)).reshape(-1, 1)
        x = np.hstack([ts, new_ID, loss])

        col_year = list(cfg.ts_vars.keys()) + ["new_ID", "loss"]
        df_year = pd.DataFrame(x, columns=col_year)
        df_year["year"] = year

        # rearrange columns according to the main df
        df_year = df_year[columns]

        # append df_year to df
        df = pd.concat([df, df_year], ignore_index=True)
        print(f"{year} {df.shape[0]}")
    print(df.head())
    print([df.loss.sum(), df.shape[0]])
    df.to_csv("synthetic_ndvi_ts.csv", index=False)
