from medil.evaluate import sfd
import numpy as np
import pandas as pd

# input
true_biadj = np.loadtxt(str(snakemake.input.true_biadj), dtype="bool", delimiter=",")
if len(true_biadj.shape) == 1:
    true_biadj = np.expand_dims(true_biadj, axis=0)

est_biadj_weights = np.loadtxt(str(snakemake.input.est_biadj_weights), delimiter=",")

cv_loss = pd.read_csv(snakemake.input.cv_loss)["avg_cv_loss"].values

losses = pd.read_csv(snakemake.input.losses)
end_losses = losses.tail(1).reset_index(drop=True)


# auto-threshold for SFD
min_sfd = 9999
for thresh in np.linspace(
    np.abs(est_biadj_weights).min(), np.abs(est_biadj_weights).max(), 20
):
    biadj_zero_pattern = (np.abs(est_biadj_weights) > thresh).astype(int)
    sfd_value, _ = sfd(biadj_zero_pattern, true_biadj)
    min_sfd = min([min_sfd, sfd_value])

# output
eval_df = pd.DataFrame(
    {
        "graph": [snakemake.wildcards["benchmark"]],
        "num_samps": [snakemake.wildcards["n"]],
        "mu": [snakemake.wildcards["mu"]],
        "lambda": [snakemake.wildcards["llambda"]],
        "cv_loss": cv_loss,
        "sfd": [min_sfd],
    }
)
eval_df = pd.concat([eval_df, end_losses], axis=1)


eval_df.to_csv(snakemake.output[0], index=False)