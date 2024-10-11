from medil.ecc_algorithms import find_heuristic_1pc
from medil.evaluate import sfd
from medil.independence_testing import estimate_UDG
import numpy as np
import pandas as pd


dataset = np.loadtxt(str(snakemake.input.dataset), delimiter=",")
true_biadj = np.loadtxt(str(snakemake.input.true_biadj), dtype="bool", delimiter=",")
if len(true_biadj.shape) == 1:
    true_biadj = np.expand_dims(true_biadj, axis=0)

udg, p_vals = estimate_UDG(dataset, method="xicor", significance_level=0.05)
est_biadj = find_heuristic_1pc(udg)

sfd_value, _ = sfd(est_biadj, true_biadj)
sfd_df = pd.DataFrame(
    {
        "alg": ["xi1pc"],
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
