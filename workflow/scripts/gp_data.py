import numpy as np
import pandas as pd
from castle.datasets.simulator import IIDSimulation, set_random_seed

def generate_interventional_gp_data(sim_gp, intervention_idx=None, intervention_value=0):
    """
    # generate interventional data for gp data
    """
    if intervention_idx is None:
        # observatioal data
        return sim_gp.X
    else:
        # original weights
        original_W = sim_gp.W.copy()
        
        # modify the weights
        num_latents = biadj_weights.shape[0]
        # set the weights of the intervention variable to 0
        sim_gp.W[:, intervention_idx] = 0
        
        # generate data
        X = sim_gp.X
        
        # set the intervention variable to the given value
        X[:, intervention_idx] = intervention_value
        
        # restore the weights
        sim_gp.W = original_W
        
        return X

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

set_random_seed(seed)
sim_gp = IIDSimulation(
    W=adj_weights,
    n=num_samps,
    method="nonlinear",
    sem_type="gp",
    noise_scale=1.0,
)

# generate data
num_latents = biadj_weights.shape[0]
datasets = []

# observational data
datasets.append(generate_interventional_gp_data(sim_gp))

# interventional data
for i in range(num_latents):
    datasets.append(generate_interventional_gp_data(sim_gp, i, 0))

for i, dataset in enumerate(datasets):
    output_path = snakemake.output.dataset.replace('.csv', f'_int{i}.csv')
    pd.DataFrame(dataset).to_csv(output_path, header=False, index=False)