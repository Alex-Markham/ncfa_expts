from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import kci
from medil.evaluate import sfd
import numpy as np
import pandas as pd


# Load the dataset
dataset = np.loadtxt(str(snakemake.input.dataset), delimiter=",")

# load true biadj for comparison
true_biadj = np.loadtxt(str(snakemake.input.true_biadj), dtype="bool", delimiter=",")
if len(true_biadj.shape) == 1:
    true_biadj = np.expand_dims(true_biadj, axis=0)

# Run FCI algorithm
g, edges = fci(
    dataset,
    kci,
    alpha=0.05,
    depth=-1,
    max_path_length=-1,
    verbose=False,
)

# Get the adjacency matrix
adj = g.graph
if len(adj.shape) == 1:
    adj = np.expand_dims(adj, axis=0)

# construct biadj
skel = adj.astype(bool)
edge_idcs = np.argwhere(np.triu(skel + skel.T))

num_latent = len(edge_idcs)
num_meas = len(skel)
est_biadj = np.zeros((num_latent, num_meas), bool)
for latent_idx, edge in enumerate(edge_idcs):
    est_biadj[latent_idx][edge] = True

sfd_value, _ = sfd(est_biadj, true_biadj)
sfd_df = pd.DataFrame(
    {
        "alg": ["fci"],
        "sfd": [sfd_value],
        "Graph": [snakemake.wildcards["benchmark"]],
        "density": [snakemake.wildcards["density"]],
        "seed": [snakemake.wildcards["seed"]],
        "num_samps": [snakemake.wildcards["n"]],
    }
)

# output
np.savetxt(snakemake.output["est_biadj"], est_biadj, delimiter=",")
sfd_df.to_csv(snakemake.output["sfd"], index=False)
