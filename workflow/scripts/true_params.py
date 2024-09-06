from medil.sample import mcm
import numpy as np


# inputs
rng = np.random.default_rng(int(snakemake.wildcards.seed))
true_biadj = np.loadtxt(str(snakemake.input.biadj), dtype="bool", delimiter=",")
if len(true_biadj.shape) == 1:
    true_biadj = np.expand_dims(true_biadj, axis=0)

# sample model parameters
true_model = mcm(rng=rng, parameterization="Gaussian", biadj=true_biadj)

# outputs
np.savetxt(snakemake.output.weights, true_model.parameters.biadj_weights, delimiter=",")
np.savetxt(
    snakemake.output.errors, true_model.parameters.error_variances, delimiter=","
)
np.savetxt(snakemake.output.means, true_model.parameters.error_means, delimiter=",")
