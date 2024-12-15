import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("oris.csv")
print(df)
df1 = df.groupby("ori_res").agg(['mean'])
df2 = df.groupby("ori_res").agg(['mean', 'std'])
print(df2.columns)
print(df2[("seed", "mean")])

# plt.errorbar(
#     df2.index,
#     df2[("runtime", "mean")],
#     yerr=df2[("runtime", "std")],
#     fmt='o',
#     capsize=5,
# )
plt.errorbar(
    df2.index,
    df2[("total_len", "mean")],
    yerr=df2[("total_len", "std")],
    fmt='o',
    capsize=5,
)
plt.ylabel("Runtime (seconds)")
plt.xlabel("Linear acceleration orientation resolution (degrees)")
plt.show()
