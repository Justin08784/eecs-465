import pandas as pd
import matplotlib.pyplot as plt

LIN_ORIS = 0
LIN_MAGS = 1
ANG_MAGS = 2
choose = LIN_ORIS

if choose == LIN_ORIS:
    df = pd.read_csv("oris.csv")
    df1 = df.groupby("ori_res").agg(['mean'])
    df2 = df.groupby("ori_res").agg(['mean', 'std'])
    plt.errorbar(
        df2.index,
        df2[("smoothness", "mean")],
        yerr=df2[("smoothness", "std")],
        fmt='o',
        capsize=5,
    )
    plt.ylabel("Runtime (seconds)")
    plt.xlabel("Linear acceleration orientation resolution (degrees)")
    plt.show()
    exit(0)

if choose == LIN_MAGS:
    df = pd.read_csv("lin_mags.csv")
    df1 = df.groupby("mag_res").agg(['mean'])
    df2 = df.groupby("mag_res").agg(['mean', 'std'])
    plt.errorbar(
        df2.index,
        df2[("smoothness", "mean")],
        yerr=df2[("smoothness", "std")],
        fmt='o',
        capsize=5,
    )
    plt.ylabel("Runtime (seconds)")
    plt.xlabel("Lin mag res (m/s)")
    plt.show()
    exit(0)

if choose == ANG_MAGS:
    df = pd.read_csv("angs.csv")
    df1 = df.groupby("ang_res").agg(['mean'])
    df2 = df.groupby("ang_res").agg(['mean', 'std'])
    plt.errorbar(
        df2.index,
        df2[("work", "mean")],
        yerr=df2[("work", "std")],
        fmt='o',
        capsize=5,
    )
    plt.ylabel("Runtime (seconds)")
    plt.xlabel("Ang res (rad/s)")
    plt.show()
 
    exit(0)

