from medil.evaluate import sfd
import numpy as np
import pandas as pd

# input
true_biadj = np.loadtxt(str(snakemake.input.true_biadj), dtype="bool", delimiter=",")
if len(true_biadj.shape) == 1:
    true_biadj = np.expand_dims(true_biadj, axis=0)

biadj = np.loadtxt(str(snakemake.input.est_biadj), delimiter=",")
cv_results = pd.read_csv(str(snakemake.input.cv_results))

llambda = float(snakemake.wildcards["llambda"])
mu = float(snakemake.wildcards["mu"])
seed = int(snakemake.wildcards["seed"])

# auto-threshold for SFD
min_sfd = 9999
for thresh in np.linspace(np.abs(biadj).min(), np.abs(biadj).max(), 20):
    biadj_zero_pattern = (np.abs(biadj) > thresh).astype(int)
    sfd_value, _ = sfd(biadj_zero_pattern, true_biadj)
    min_sfd = min([min_sfd, sfd_value])

# output
eval_df = pd.DataFrame(
    {
        "elbo cross validation": cv_results["cv_loss"].mean(),
        "sfd": [min_sfd],
        "Graph": [snakemake.wildcards["idx"]],
        r"$\lambda$": [llambda],
        r"$\mu$": [mu],
        "seed": [seed],
    }
)
eval_df.to_csv(snakemake.output["eval"], index=False)