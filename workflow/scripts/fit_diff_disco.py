!pip install ananke-causal

import numpy as np
import pandas as pd

from dcd.admg_discovery import Discovery
from utils.admg2pag import admg_to_pag

def run_admg_discovery(dataset_path, admg_classes=["bowfree", "ancestral", "arid"], lamda=0.05, verbose=True):
    """
    Run ADMG discovery on the given dataset

    """
    # Load the dataset
    data = np.loadtxt(dataset_path, delimiter=",")

    # Create dataframe
    column_names = [f'V{i+1}' for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=column_names)

    results = {}
    for admg_class in admg_classes:
        if verbose:
            print(f"Running discovery for ADMG class: {admg_class}")

        learn = Discovery(lamda=lamda)
        best_G = learn.discover_admg(df, admg_class=admg_class, verbose=verbose)

        results[admg_class] = {
            'convergence': learn.convergence_,
            'di_edges': best_G.di_edges,
            'bi_edges': best_G.bi_edges, # biadj
            'pag': admg_to_pag(best_G)
        }

    return results[admg_class]['bi_edges']

results = run_admg_discovery()
