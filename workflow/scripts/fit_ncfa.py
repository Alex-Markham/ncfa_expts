from medil.evaluate import sfd
from medil.models import NeuroCausalFactorAnalysis
import numpy as np
import pandas as pd
import torch


# input
threshold = 0.5
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

ncfa.fit(dataset)

biadj = ncfa.parameters.weights

# figure out threshold or auto-threshold for SFD
biadj_zero_pattern = (np.abs(biadj) > threshold).astype(int)
sfd_value, ushd_value = sfd(biadj_zero_pattern, true_biadj)

# output
eval_df = pd.DataFrame(
    {
        "elbo validation": ncfa.loss["elbo_valid"],
        "sfd": [sfd_value],
        "Graph": [snakemake.wildcards["idx"]],
        r"$\lambda$": [llambda],
        r"$\mu$": [mu],
        "seed": [seed],
    }
)
eval_df.to_csv(snakemake.output["eval"], index=False)

np.savetxt(snakemake.output["biadj"], biadj, delimiter=",")
