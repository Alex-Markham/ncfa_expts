import numpy as np
from scipy.stats import bernoulli

# manually specified benchmark graphs
biadj_dict = {
    0: np.array([[1, 1]], dtype=bool),
    1: np.array([[1, 1, 1]], dtype=bool),
    2: np.array([[1, 1, 1, 1]], dtype=bool),
    3: np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    4: np.array([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]], dtype=bool),
    5: np.array(
        [[1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1]],
        dtype=bool,
    ),
    # 6: np.array(
    #     [
    #         [1, 1, 1, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 1, 1, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 1, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 1, 1, 1],
    #     ],
    #     dtype=bool,
    # ),
    # 7: np.array(
    #     [
    #         [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    #     ],
    #     dtype=bool,
    # ),
    6: np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 1],
        ],
        dtype=bool,
    ),  # 8:
}

# generate random biadjacency matrix
def generate_random_biadj(n_latent, n_observed, sparsity, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Bernoulli distribution
    random_biadj = bernoulli.rvs(1 - sparsity, size=(n_latent, n_observed))

    # ensure at least one observed variable is connected to each latent variable
    for col in range(n_observed):
        if np.sum(random_biadj[:, col]) == 0:
            random_biadj[np.random.randint(n_latent), col] = 1
    
    return random_biadj.astype(bool)

idx = int(snakemake.wildcards.idx)
sparsity = float(snakemake.params.get("sparsity", 0.5))  
n_latent = int(snakemake.params.get("n_latent", 3))  
n_observed = int(snakemake.params.get("n_observed", 5))  
seed = int(snakemake.params.get("seed", 42))  

# select one graph
#biadj = biadj_dict[int(snakemake.wildcards.idx)]
if idx in biadj_dict:
    biadj = biadj_dict[idx]
else:
    biadj = generate_random_biadj(n_latent, n_observed, sparsity, seed)

# output
np.savetxt(snakemake.output.biadj, biadj, delimiter=",", fmt="%1u")
