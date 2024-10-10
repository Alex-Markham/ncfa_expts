from medil.models import GaussianMCM
import numpy as np


# inputs
weights = np.loadtxt(snakemake.input.weights, delimiter=",")
if len(weights.shape) == 1:
    weights = np.expand_dims(weights, axis=0)
means = np.loadtxt(snakemake.input.means, delimiter=",")
errors = np.loadtxt(snakemake.input.errors, delimiter=",")

# load model and sample data
mcm = GaussianMCM(biadj=weights)
params = mcm.parameters
params.biadj_weights = weights
params.error_means = means
params.error_variances = errors

dataset = mcm.sample(int(snakemake.wildcards["n"]))

# output
np.savetxt(snakemake.output.dataset, dataset, delimiter=",")
