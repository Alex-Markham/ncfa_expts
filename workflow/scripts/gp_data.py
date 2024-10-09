import numpy as np
import pandas as pd
from castle.datasets.simulator import IIDSimulation, set_random_seed


def biadj_to_adj(biadj):
    num_latent, num_meas = biadj.shape
    adj = np.zeros((num_latent + num_meas, num_latent + num_meas), dtype=biadj.dtype)
    adj[:num_latent, num_latent:] = biadj
    return adj


biadj_weights_path = snakemake.input.weights

num_samps = int(snakemake.wildcards.n)
seed = int(snakemake.wildcards.seed)

biadj_weights = pd.read_csv(biadj_weights_path).values
adj_weights = biadj_to_adj(biadj_weights)

# gp sampling
set_random_seed(seed)
sim_gp = IIDSimulation(
    W=adj_weights,
    n=num_samps,
    method="nonlinear",
    sem_type="gp",
    noise_scale=1.0,
)
X_gp = sim_gp.X

pd.DataFrame(X_gp).to_csv(snakemake.output.dataset, header=False, index=False)
