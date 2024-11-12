from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


# input
plot_data = pd.read_csv(snakemake.input[0])
p1, p2 = plot_data[r"mu"].iloc[0:2]
side_length = p2 - p1


# heatmap drawer for facetgrid
def draw_heatmap(*args, **kwargs):
    data = kwargs.pop("data")

    # given best sfd, find best elbo validation
    sfds = data[data["loss type"] == "SFD"]
    best_sfds = sfds[sfds["loss"] == sfds["loss"].min()]
    best_elbo_val, best_lambda, best_mu = np.inf, None, None
    elbo_vals = data["loss type"] == "elbo_valid"
    for llambda, mu in best_sfds[["lambda", "mu"]].values:
        elbo_val = elbo_vals[elbo_vals["lambda"] == llambda & elbo_vals["mu"] == mu][
            "loss"
        ]
        if elbo_val < best_elbo_val:
            best_elbo_val = elbo_val
            best_lambda = llambda
            best_mu = mu

    # select lowest CV loss
    selected_params = data.iloc[data["cv_loss"].argmin()]
    selected_lambda = selected_params[r"lambda"]
    selected_mu = selected_params[r"mu"]

    data = data.round({r"lambda": 2, r"mu": 2})
    d = data.pivot(index=args[0], columns=args[1], values=args[2])
    h = sns.heatmap(
        d, cbar=False, square=True, annot=True, annot_kws={"size": 5}, **kwargs
    )
    h.invert_yaxis()
    c = d.columns

    best_lambda_idx = np.argwhere(c == best_lambda)[0][0]
    best_mu_idx = np.argwhere(c == best_mu)[0][0]
    h.add_patch(
        Rectangle(
            (best_lambda_idx, best_mu_idx),
            1,
            1,
            fill=False,
            edgecolor="red",
            lw=3,
        )
    )

    selected_lambda_idx = np.argwhere(c == selected_lambda)[0][0]
    selected_mu_idx = np.argwhere(c == selected_mu)[0][0]
    h.add_patch(
        Rectangle(
            (selected_lambda_idx, selected_mu_idx),
            1,
            1,
            fill=False,
            edgecolor="green",
            lw=3,
        )
    )


## plot
sfd_df = plot_data.rename(columns={"sfd": "loss"})
sfd_df["loss type"] = "SFD"

shd_df = plot_data.rename(columns={"shd": "loss"})
shd_df["loss type"] = "SHD"

elbo_train_df = plot_data.rename(columns={"elbo_train": "loss"})
elbo_train_df["loss type"] = "elbo train"

elbo_valid_df = plot_data.rename(columns={"elbo_valid": "loss"})
elbo_valid_df["loss type"] = "elbo valid"

recon_train_df = plot_data.rename(columns={"recon_train": "loss"})
recon_train_df["loss type"] = "recon train"

recon_valid_df = plot_data.rename(columns={"recon_valid": "loss"})
recon_valid_df["loss type"] = "recon valid"

plot_df = pd.concat(
    [sfd_df, shd_df, elbo_train_df, elbo_valid_df, recon_train_df, recon_valid_df]
).round({r"lambda": 2, r"mu": 2})

m = sns.FacetGrid(plot_df, row="graph", col="loss type", margin_titles=True)
m.map_dataframe(draw_heatmap, r"lambda", r"mu", "loss")

m.fig.subplots_adjust(top=0.9)
m.fig.suptitle("Regularized NCFA")

m.figure.tight_layout()


# output
m.figure.savefig(snakemake.output[0])
