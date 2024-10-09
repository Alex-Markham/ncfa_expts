from medil.sample import biadj
import numpy as np

# manually specified benchmark graphs
biadj_dict = {
    "0": np.array([[1, 1]], dtype=bool),
    "1": np.array([[1, 1, 1]], dtype=bool),
    "2": np.array([[1, 1, 1, 1]], dtype=bool),
    "3": np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    "4": np.array([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]], dtype=bool),
    "5": np.array(
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
    "6": np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 1],
        ],
        dtype=bool,
    ),  # 8:
}


benchmark = str(snakemake.wildcards.benchmark)

# select one graph from the benchmark graphs or generate a random one
if benchmark in biadj_dict:
    biadj_matrix = biadj_dict[benchmark]
elif benchmark == "random":
    density = float(snakemake.wildcards.density)
    num_latent = int(snakemake.wildcards.num_latent)
    num_meas = int(snakemake.wildcards.num_meas)
    seed = int(snakemake.wildcards.seed)

    rng = np.random.default_rng(seed)
    biadj_matrix = biadj(
        num_meas=num_meas,
        density=density,
        one_pure_child=True,
        num_latent=num_latent,
        rng=rng,
    )
else:
    raise ValueError(
        "`index` must either be 'random' or corresponding to the manually specified graphs in `workflow/scripts/true_biadj.py`"
    )

# output
np.savetxt(snakemake.output.biadj, biadj_matrix, delimiter=",", fmt="%1u")
