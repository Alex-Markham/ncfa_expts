from medil.models import GaussianMCM
import numpy as np

def generate_interventional_data(mcm, intervention_idx=None, intervention_value=0):
    """
    Generate observational or interventional data from a GaussianMCM model.
    Args:
        mcm: GaussianMCM model
        intervention_idx: index of latent variable to be intervened, 'None' means no intervention (observational data)
        intervention_value: intervention value, default is 0
    """
    if intervention_idx is None:
        return mcm.sample(int(snakemake.wildcards["n"]))
    else:
        # generate interventional data
        # preserve original parameters
        original_mean = mcm.parameters.error_means[intervention_idx]
        original_variance = mcm.parameters.error_variances[intervention_idx]
        
        # set intervention value
        mcm.parameters.error_means[intervention_idx] = intervention_value
        mcm.parameters.error_variances[intervention_idx] = 0
        
        # generate data
        dataset = mcm.sample(int(snakemake.wildcards["n"]))
        
        # restore original parameters
        mcm.parameters.error_means[intervention_idx] = original_mean
        mcm.parameters.error_variances[intervention_idx] = original_variance
        
        return dataset

weights = np.loadtxt(snakemake.input.weights, delimiter=",")
if len(weights.shape) == 1:
    weights = np.expand_dims(weights, axis=0)
means = np.loadtxt(snakemake.input.means, delimiter=",")
errors = np.loadtxt(snakemake.input.errors, delimiter=",")

mcm = GaussianMCM(biadj=weights)
params = mcm.parameters
params.biadj_weights = weights
params.error_means = means
params.error_variances = errors

# generate observational data
num_latents = weights.shape[0]
datasets = []
datasets.append(generate_interventional_data(mcm))

# generate interventional data
for i in range(num_latents):
    datasets.append(generate_interventional_data(mcm, i, 0))

for i, dataset in enumerate(datasets):
    output_path = snakemake.output.dataset.replace('.csv', f'_int{i}.csv')
    np.savetxt(output_path, dataset, delimiter=",")