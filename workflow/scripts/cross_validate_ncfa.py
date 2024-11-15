from medil.models import NeuroCausalFactorAnalysis
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch

# input
k = 10
dataset = np.loadtxt(str(snakemake.input.dataset), delimiter=",")
llambda = float(snakemake.wildcards["llambda"])
mu = float(snakemake.wildcards["mu"])
seed = int(snakemake.wildcards["seed"])

np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

kf = KFold(
    n_splits=k,
    shuffle=True,
    random_state=seed,
)

# k-folds cross validation
cv_losses = np.empty(k, float)
for idx, split_idcs in enumerate(kf.split(dataset)):
    split_ncfa = NeuroCausalFactorAnalysis(seed=seed)
    split_ncfa.hyperparams["mu"] = mu
    split_ncfa.hyperparams["lambda"] = llambda
    split_ncfa.fit(dataset, split_idcs)
    cv_losses[idx] = split_ncfa.loss["elbo_valid"][-1]

# output
avg_cv_loss = cv_losses.mean()

cv_df = pd.DataFrame(
    {
        "lambda": [llambda],
        "mu": [mu],
        "avg_cv_loss": [cv_losses.mean()],
    }
)

cv_df.to_csv(snakemake.output["avg_cv_loss"], index=False)
