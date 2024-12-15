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
        df2[("runtime", "mean")],
        yerr=df2[("runtime", "std")],
        fmt='o',
        capsize=5,
    )
    plt.ylabel("Runtime (seconds)")
    plt.xlabel("Linear acceleration orientation resolution (degrees)")
    plt.show()
    exit(0)

if choose == LIN_MAGS:
    print("dipshit")
    exit(0)

if choose == ANG_MAGS:
    exit(0)

