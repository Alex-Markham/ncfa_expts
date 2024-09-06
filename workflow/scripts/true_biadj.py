import numpy as np


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

# select one graph
biadj = biadj_dict[int(snakemake.wildcards.idx)]

# output
np.savetxt(snakemake.output.biadj, biadj, delimiter=",", fmt="%1u")
