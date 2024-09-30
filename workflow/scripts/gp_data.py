import numpy as np
import pandas as pd
from castle.datasets.simulator import IIDSimulation
import networkx as nx

def biadj_to_adj(biadj):
    num_latent, num_meas = biadj.shape
    adj = np.zeros((num_latent + num_meas, num_latent + num_meas), dtype=biadj.dtype)
    adj[:num_latent, num_latent:] = biadj
    return adj

if isinstance(snakemake.input.biadj, list):
    biadj_file = snakemake.input.biadj[0]
else:
    biadj_file = str(snakemake.input.biadj)

#biadj_file = snakemake.input.biadj

output_file = snakemake.output.dataset
n_samples = int(snakemake.wildcards.n)
seed = int(snakemake.wildcards.seed)

biadj_mat = pd.read_csv(biadj_file, index_col=0).values.astype(bool)
adj_matrix = biadj_to_adj(biadj_mat)

# check DAG
G_nx = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
if not nx.is_directed_acyclic_graph(G_nx):
    raise ValueError("The input graph is not a DAG")

rng = np.random.default_rng(seed=seed)
num_edges = int(biadj_mat.sum())
weights = (rng.random(num_edges) * 1.5) + 0.5
sign_choices = rng.choice([True, False], size=num_edges)
weights[sign_choices] *= -1

biadj_weights = np.zeros_like(biadj_mat, dtype=float)
biadj_weights[biadj_mat] = weights
adj_matrix_weighted = biadj_to_adj(biadj_weights)

# gp sampling
sim_gp = IIDSimulation(
    W=adj_matrix_weighted,
    n=n_samples,
    method='nonlinear',
    sem_type='gp',
    noise_scale=1.0
)
X_gp = sim_gp.X

pd.DataFrame(X_gp).to_csv(output_file, index=False)