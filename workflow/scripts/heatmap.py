from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


# input
plot_data = pd.read_csv(snakemake.input[0])
p1, p2 = plot_data[r"$\mu$"].iloc[0:2]
side_length = p2 - p1


# heatmap drawer for facetgrid
def draw_heatmap(*args, **kwargs):
    data = kwargs.pop("data")
    best_row = data.iloc[data["elbo validation"].argmin()]
    best_mu = best_row[r"$\mu$"]
    best_lambda = best_row[r"$\lambda$"]
    data = data.round({r"$\lambda$": 2, r"$\mu$": 2})
    d = data.pivot(index=args[0], columns=args[1], values=args[2])
    h = sns.heatmap(
        d, cbar=False, square=True, annot=True, annot_kws={"size": 5}, **kwargs
    )
    h.invert_yaxis()
    c = d.columns
    best_mu_idx = np.argwhere(c == best_mu)[0][0]
    best_lambda_idx = np.argwhere(c == best_lambda)[0][0]
    h.add_patch(
        Rectangle(
            (best_mu_idx, best_lambda_idx),
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

l2_df = plot_data.rename(columns={"recon loss": "loss"})
l2_df["loss type"] = "recon train"

mle_df = pd.concat([sfd_df, l2_df]).round({r"$\lambda$": 2, r"$\mu$": 2})

m = sns.FacetGrid(mle_df, row="Graph", col="loss type", margin_titles=True)
m.map_dataframe(draw_heatmap, r"$\lambda$", r"$\mu$", "loss")

m.fig.subplots_adjust(top=0.9)
m.fig.suptitle("Regularized NCFA")

m.figure.tight_layout()


# output
m.figure.savefig(snakemake.output[0])
