from medil.models import NeuroCausalFactorAnalysis
import numpy as np
import pandas as pd
import torch

dataset = np.loadtxt(str(snakemake.input.dataset), delimiter=",")
llambda = float(snakemake.wildcards["llambda"])
mu = float(snakemake.wildcards["mu"])
seed = int(snakemake.wildcards["seed"])


torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

# fit to full dataset
ncfa = NeuroCausalFactorAnalysis(seed=seed)
ncfa.hyperparams["mu"] = mu
ncfa.hyperparams["lambda"] = llambda

no_split = (range(len(dataset)), [0, 1])
ncfa.fit(dataset, no_split)

pooled_biadj_weights = ncfa.parameters.biadj

# output
np.savetxt(
    snakemake.output["pooled_biadj_weights"], pooled_biadj_weights, delimiter=","
)
torch.save(ncfa.parameters.vae, snakemake.output["vae"])
pd.DataFrame(ncfa.loss).to_csv(snakemake.output["losses"], index=False)
