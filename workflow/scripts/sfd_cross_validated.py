from medil.evaluate import sfd
import numpy as np
import pandas as pd


true_biadj = np.loadtxt(str(snakemake.input.true_biadj), dtype="bool", delimiter=",")
if len(true_biadj.shape) == 1:
    true_biadj = np.expand_dims(true_biadj, axis=0)

est_biadj_weights = np.loadtxt(str(snakemake.input.est_biadj_weights), delimiter=",")

# auto-threshold for SFD
min_sfd = 9999
for thresh in np.linspace(
    np.abs(est_biadj_weights).min(), np.abs(est_biadj_weights).max(), 20
):
    biadj_zero_pattern = (np.abs(est_biadj_weights) > thresh).astype(int)
    sfd_value, _ = sfd(biadj_zero_pattern, true_biadj)
    min_sfd = min([min_sfd, sfd_value])

# output
sfd_df = pd.DataFrame(
    {
        "sfd": [min_sfd],
        "Graph": [snakemake.wildcards["benchmark"]],
        "density": [snakemake.wildcards["density"]],
        "seed": [snakemake.wildcards["seed"]],
        "num_samps": [snakemake.wildcards["n"]],
    }
)
sfd_df.to_csv(snakemake.output["sfd"], index=False)
