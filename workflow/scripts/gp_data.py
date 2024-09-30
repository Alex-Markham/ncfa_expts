import numpy as np
import pandas as pd
from castle.datasets.simulator import IIDSimulation


def biadj_to_adj(biadj):
    num_latent, num_meas = biadj.shape
    adj = np.zeros((num_latent + num_meas, num_latent + num_meas), dtype=biadj.dtype)
    adj[:num_latent, num_latent:] = biadj
    return adj


biadj_path = snakemake.input.biadj[0]

# biadj_file = snakemake.input.biadj

output_file = snakemake.output.dataset
n_samples = int(snakemake.wildcards.n)
seed = int(snakemake.wildcards.seed)

biadj_mat = pd.read_csv(biadj_path, index_col=0).values.astype(bool)
adj_matrix = biadj_to_adj(biadj_mat)

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
    method="nonlinear",
    sem_type="gp",
    noise_scale=1.0,
)
X_gp = sim_gp.X

pd.DataFrame(X_gp).to_csv(output_file, header=False, index=False)
