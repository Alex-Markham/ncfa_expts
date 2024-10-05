import numpy as np
import pandas as pd

def sfd(predicted_biadj, true_biadj):
    """Perform analysis of the distances between true and reconstructed structures
    Parameters
    ----------
    biadj_mat: input directed graph
    biadj_mat_recon: learned directed graph in the form of adjacency matrix

    Returns
    -------
    sfd: squared Frobenius distance (bipartite graph)
    ushd: structural hamming distance (undirected graph)
    """

    # ushd = shd_func(recover_ug(biadj_mat), recover_ug(biadj_mat_recon))
    ug = recover_ug(true_biadj)
    ug_recon = recover_ug(predicted_biadj)

    ushd = np.triu(np.logical_xor(ug, ug_recon), 1).sum()

    true_biadj = true_biadj.astype(int)
    predicted_biadj = predicted_biadj.astype(int)

    wtd_ug = true_biadj.T @ true_biadj
    wtd_ug_recon = predicted_biadj.T @ predicted_biadj

    sfd = ((wtd_ug - wtd_ug_recon) ** 2).sum()

    return sfd, ushd

def recover_ug(biadj_mat):
    """Recover the undirected graph from the directed graph
    Parameters
    ----------
    biadj_mat: learned directed graph

    Returns
    -------
    ug: the recovered undirected graph
    """
    # get the undirected graph from the directed graph
    ug = biadj_mat.T @ biadj_mat
    np.fill_diagonal(ug, False)

    return ug

def main(true_graph_file, est_graph_file, output_file):
    true_graph = pd.read_csv(true_graph_file, header=None).values
    est_graph = pd.read_csv(est_graph_file, header=None).values

    print(f"True graph shape: {true_graph.shape}")
    print(f"Estimated graph shape: {est_graph.shape}")

    # make sure the graphs have the same shape
    if true_graph.shape != est_graph.shape:
        max_rows = max(true_graph.shape[0], est_graph.shape[0])
        max_cols = max(true_graph.shape[1], est_graph.shape[1])
        
        true_graph_padded = np.zeros((max_rows, max_cols))
        true_graph_padded[:true_graph.shape[0], :true_graph.shape[1]] = true_graph
        
        est_graph_padded = np.zeros((max_rows, max_cols))
        est_graph_padded[:est_graph.shape[0], :est_graph.shape[1]] = est_graph
        
        true_graph = true_graph_padded
        est_graph = est_graph_padded

    sfd_value, ushd_value = sfd(est_graph, true_graph)

    result = pd.DataFrame({
        'idx': [snakemake.wildcards.idx],
        'seed': [snakemake.wildcards.seed],
        'n': [snakemake.wildcards.n],
        'lambda': [snakemake.wildcards.lambda_],
        'mu': [snakemake.wildcards.mu],
        'sfd': [sfd_value],
        'ushd': [ushd_value]
    })

    result.to_csv(output_file, index=False)

if __name__ == "__main__":
    true_graph_file = snakemake.input.true_graph
    est_graph_file = snakemake.input.est_graph
    output_file = snakemake.output[0]
    main(true_graph_file, est_graph_file, output_file)