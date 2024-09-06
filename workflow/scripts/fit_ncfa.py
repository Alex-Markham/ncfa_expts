from medil.evaluate import sfd, min_perm_squared_l2_dist_abs
from medil.models import DevMedil
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import KFold


# input
threshold = 0.5
k = 10
dataset = np.loadtxt(str(snakemake.input.dataset), delimiter=",")
true_biadj = np.loadtxt(str(snakemake.input.true_biadj), dtype="bool", delimiter=",")
if len(true_biadj.shape) == 1:
    true_biadj = np.expand_dims(true_biadj, axis=0)
llambda = float(snakemake.wildcards["llambda"])
mu = float(snakemake.wildcards["mu"])
method = snakemake.wildcards["method"]
seed = int(snakemake.wildcards["seed"])

weights = np.loadtxt(snakemake.input.true_params[0], delimiter=",")
if len(weights.shape) == 1:
    weights = np.expand_dims(weights, axis=0)
true_mean = np.loadtxt(snakemake.input.true_params[1], delimiter=",")
errors = np.loadtxt(snakemake.input.true_params[2], delimiter=",")


def rng():
    return np.random.default_rng(seed)


# fit to full dataset to get sfd and l2
full_model = DevMedil(rng=rng())
full_model.fit(dataset, method=method, lambda_reg=llambda, mu_reg=mu)

w_hat = full_model.W_hat
w_hat_zero_pattern = (np.abs(w_hat) > threshold).astype(int)
sfd_value, ushd_value = sfd(w_hat_zero_pattern, true_biadj)

true_sigma = weights.T @ weights + np.diag(errors)

est_sigma = w_hat.T @ w_hat + full_model.D_hat
est_mean = dataset.mean(0)

l2 = ((est_mean - true_mean) ** 2).sum() + ((est_sigma - true_sigma) ** 2).sum()

kf = KFold(
    n_splits=k,
    shuffle=True,
    random_state=seed,
)

# k-folds cross validation
cv_losses = np.empty(k, float)
for idx, (train_index, val_index) in enumerate(kf.split(dataset)):
    train_data = dataset[train_index]
    val_data = dataset[val_index]

    split_model = DevMedil(rng=rng())
    split_model.fit(train_data, method=method, lambda_reg=llambda, mu_reg=mu)
    if method == "lse":
        cv_loss = split_model.validation_lse(0, 0, val_data)
    elif method == "mle":
        cv_loss = split_model.validation_mle(0, 0, val_data)
    cv_losses[idx] = cv_loss

# output
eval_df = pd.DataFrame(
    {
        "cross validation loss": cv_losses.mean(),
        "sfd": [sfd_value],
        r"$l_2$": [l2],
        "Graph": [snakemake.wildcards["idx"]],
        r"$\lambda$": [llambda],
        r"$\mu$": [mu],
        "seed": [seed],
        "method": [method],
    }
)
eval_df.to_csv(snakemake.output["eval"], index=False)

np.savetxt(snakemake.output["weights"], w_hat, delimiter=",")
np.savetxt(snakemake.output["errors"], full_model.D_hat, delimiter=",")
np.savetxt(snakemake.output["means"], est_mean, delimiter=",")
