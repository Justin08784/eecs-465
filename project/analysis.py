import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

LIN_ORIS = 0
LIN_MAGS = 1
ANG_MAGS = 2
choose = LIN_ORIS

if choose == LIN_ORIS:

    num_primitives = np.log(np.array([873, 441, 85, 65, 15, 9]))

    # Interesting:
    # - runtime / num_targets
    # - work / total_len
    # - ** num_nodes / num_targets
    # - ** runtime
    # - ** runtime / num_nodes
    # - ** total_delta_ang / num_nodes
    df = pd.read_csv("combined2.csv")
    df2 = df.groupby("ori_res").agg(['mean', 'std'])
    df2[("work", "mean")] /= df2[("total_len", "mean")]

    df2["num_primitives"] = num_primitives
    plt.errorbar(
        df2["num_primitives"],
        df2[("num_targets", "mean")],
        yerr=df2[("num_targets", "std")],
        fmt='o',
        capsize=5,
    )
    plt.ylabel("Number of targets")
    plt.xlabel("Log number of primitives")
    plt.show()
    print(df2)
    df2.to_csv("jawohl.csv")
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

