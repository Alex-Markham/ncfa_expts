import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sfd_results = pd.read_csv(snakemake.input[0])

plt.figure(figsize=(12, 6))
sns.boxplot(x="density", y="sfd", data=sfd_results, hue="alg")
plt.title("SFD between True Graph and NCFA-Learned Structure")
plt.xlabel("graph density")
plt.ylabel("Structural Frobenius Difference (SFD)")

plt.savefig(snakemake.output[0])
