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
    best_row = data.iloc[data["cross validation loss"].argmin()]
    best_mu = best_row[r"$\mu$"]
    best_lambda = best_row[r"$\lambda$"]
    data = data.round({r"$\lambda$": 2, r"$\mu$": 2})
    d = data.pivot(index=args[0], columns=args[1], values=args[2])
    h = sns.heatmap(
        d, cbar=False, square=True, annot=True, annot_kws={"size": 5}, **kwargs
    )
    h.invert_yaxis()
    x = best_mu + 0.5 if best_mu > 0.5 else best_mu
    y = best_lambda + 0.5 if best_lambda > 0.5 else best_lambda
    h.add_patch(
        Rectangle(
            (x, y),
            1,
            1,
            fill=False,
            edgecolor="green",
            lw=3,
        )
    )


## MLE
mle_df = plot_data[plot_data["method"] == "mle"]

sfd_df = pd.DataFrame(mle_df).rename(columns={"sfd": "loss"})
sfd_df["loss type"] = "sfd"

l2_df = pd.DataFrame(mle_df).rename(columns={"$l_2$": "loss"})
l2_df["loss type"] = "$l_2$"

mle_df = pd.concat([sfd_df, l2_df]).round({r"$\lambda$": 2, r"$\mu$": 2})

m = sns.FacetGrid(mle_df, row="Graph", col="loss type", margin_titles=True)
m.map_dataframe(draw_heatmap, r"$\lambda$", r"$\mu$", "loss")

m.fig.subplots_adjust(top=0.9)
m.fig.suptitle("MLE")

m.figure.tight_layout()


## LSE
lse_df = plot_data[plot_data["method"] == "lse"]

sfd_df = pd.DataFrame(lse_df).rename(columns={"sfd": "loss"})
sfd_df["loss type"] = "sfd"

l2_df = pd.DataFrame(lse_df).rename(columns={"$l_2$": "loss"})
l2_df["loss type"] = "$l_2$"

lse_df = pd.concat([sfd_df, l2_df])

l = sns.FacetGrid(lse_df, row="Graph", col="loss type", margin_titles=True)
l.map_dataframe(draw_heatmap, r"$\lambda$", r"$\mu$", "loss")

l.fig.subplots_adjust(top=0.9)
l.fig.suptitle("LSE")

l.figure.tight_layout()

# output
# output
m.figure.savefig(snakemake.output["mle"])
l.figure.savefig(snakemake.output["lse"])
