from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import kci
from causallearn.utils.GraphUtils import GraphUtils
import numpy as np


# Load the dataset
data = np.loadtxt(dataset_path, delimiter=",")

# Run FCI algorithm
g, edges = fci(
    data,
    kci,
    alpha=0.05,
    depth=-1,
    max_path_length=-1,
    verbose=False,
)

# Get the adjacency matrix
adj_matrix = g.graph.graph

# Save the output, adjacency matrix
np.savetxt(output_path, adj_matrix, delimiter=",")

# Interpret edge properties
edge_interpretations = []
for edge in edges:
    interpretation = f"Edge {edge.get_node1()} - {edge.get_node2()}: "
    if "nl" in edge.properties:
        interpretation += "No latent confounder. "
    else:
        interpretation += "Possibly latent confounders. "
    if "dd" in edge.properties:
        interpretation += "Definitely direct. "
    elif "pd" in edge.properties:
        interpretation += "Possibly direct. "
    else:
        interpretation += "Not direct. "

    edge_interpretations.append(interpretation)

# Save edge interpretations
with open("edge_interpretations.txt", "w") as f:
    for interp in edge_interpretations:
        f.write(interp + "\n")

# return g, edges, adj_matrix
