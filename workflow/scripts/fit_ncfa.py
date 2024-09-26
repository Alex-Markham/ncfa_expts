from medil.evaluate import sfd
from medil.models import NeuroCausalFactorAnalysis
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch


# input
k = 10
# threshold = 0.5 # using auto-thresh instead
dataset = np.loadtxt(str(snakemake.input.dataset), delimiter=",")
true_biadj = np.loadtxt(str(snakemake.input.true_biadj), dtype="bool", delimiter=",")
if len(true_biadj.shape) == 1:
    true_biadj = np.expand_dims(true_biadj, axis=0)
llambda = float(snakemake.wildcards["llambda"])
mu = float(snakemake.wildcards["mu"])
seed = int(snakemake.wildcards["seed"])


def rng():
    return np.random.default_rng(seed)


np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

# fit to full dataset to get sfd and l2
ncfa = NeuroCausalFactorAnalysis()
ncfa.hyperparams["mu"] = mu
ncfa.hyperparams["lambda"] = llambda

no_split = (range(len(dataset)), [0, 1])
ncfa.fit(dataset, no_split)

biadj = ncfa.parameters.weights

# auto-threshold for SFD
min_sfd = 9999
for thresh in np.linspace(np.abs(biadj).min(), np.abs(biadj).max(), 20):
    biadj_zero_pattern = (np.abs(biadj) > thresh).astype(int)
    sfd_value, ushd_value = sfd(biadj_zero_pattern, true_biadj)
    min_sfd = min([min_sfd, sfd_value])

kf = KFold(
    n_splits=k,
    shuffle=True,
    random_state=seed,
)
# k-folds cross validation
cv_losses = np.empty(k, float)
for idx, split_idcs in enumerate(kf.split(dataset)):
    split_ncfa = NeuroCausalFactorAnalysis()
    split_ncfa.hyperparams["mu"] = mu
    split_ncfa.hyperparams["lambda"] = llambda
    split_ncfa.fit(dataset, split_idcs)
    cv_losses[idx] = ncfa.loss["elbo_valid"][-1]


# output
eval_df = pd.DataFrame(
    {
        "elbo cross validation": cv_losses.mean(),
        "recon loss": ncfa.loss["recon_train"][-1],
        "sfd": [min_sfd],
        "Graph": [snakemake.wildcards["idx"]],
        r"$\lambda$": [llambda],
        r"$\mu$": [mu],
        "seed": [seed],
    }
)
eval_df.to_csv(snakemake.output["eval"], index=False)

np.savetxt(snakemake.output["biadj"], biadj, delimiter=",")
