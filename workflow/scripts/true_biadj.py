import numpy as np
from numpy.random import Generator, default_rng
import numpy.typing as npt

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
def biadj(
    num_meas: int,
    density: float = 0.2,
    one_pure_child: bool = True,
    num_latent: int = 0,
    rng: Generator = default_rng(),
) -> npt.NDArray:
    """Randomly generate biadjacency matrix for graphical minMCM."""
    if one_pure_child:
        """Define a maximum independent set of size `num_latent`, and then grow these into a minimum edge clique cover with average max clique size `2 + (num_meas - num_latent) * density`."""
        if num_latent == 0:
            num_latent = rng.integers(1, num_meas)
        if density is None:
            density = rng.random()

        # specify pure children/independent set
        biadj = np.zeros((num_latent, num_meas), bool)
        biadj[:, :num_latent] = np.eye(num_latent)

        # every child gets a parent; specifically L_0, until the
        # within-column perm below using np.permuted
        biadj[0, num_latent:] = True

        # randomly fill in remaining density * (num_meas - num_latent)
        # * (num_latent - 1) edges
        max_num_edges = (num_meas - num_latent) * (num_latent - 1)
        num_edges = np.round(max_num_edges * density).astype(int)

        edges = np.zeros(max_num_edges, bool)
        edges[:num_edges] = True
        edges = rng.permutation(edges).reshape(num_latent - 1, num_meas - num_latent)

        biadj[1:][:, num_latent:] = edges

        nonpure_children = biadj[:, num_latent:]
        biadj[:, num_latent:] = rng.permuted(nonpure_children, axis=0)

        # change child order, so pure children aren't first
        biadj = rng.permutation(biadj, axis=1)

    else:
        """Generate minMCM from Erdős–Rényi random undirected graph
        over observed variables."""
        if num_latent != 0:
            msg = "`num_latent` can only be specified when `one_pure_child==True`."
            raise ValueError(msg)

        udg = np.zeros((num_meas, num_meas), bool)

        max_edges = (num_meas * (num_meas - 1)) // 2
        num_edges = np.round(density * max_edges).astype(int)

        edges = np.ones(max_edges)
        edges[num_edges:] = 0

        udg[np.triu_indices(num_meas, k=1)] = rng.permutation(edges)
        udg += udg.T
        np.fill_diagonal(udg, True)

        # find latent connections (minimum edge clique cover)
        biadj = find_clique_min_cover(udg).astype(bool)

    return biadj

# copy paste the function from ecc_algorithms
import subprocess, os, shutil
from collections import deque

def find_clique_min_cover(graph, verbose=False):
    """Returns the clique-minimum edge clique cover.

    Parameters
    ----------
    graph : np.array
            Adjacency matrix for undirected graph.

    verbose : bool, optional
              Wether or not to print verbose output.

    Returns
    -------
    the_cover: np.array
               Biadjacency matrix representing edge clique cover.

    See Also
    --------
    graph.UndirectedDependenceGraph : Defines auxilliary data structure
                                      and reduction rules used by this
                                      algorithm.

    Notes
    -----
    This is an implementation of the algorithm described in
    :cite:`Gramm_2009`.

    """
    graph = UndirectedDependenceGraph(graph, verbose)
    try:
        graph.make_aux()
    except ValueError:
        print("The input graph doesn't appear to have any edges!")
        return graph.adj_matrix

    num_cliques = 1
    the_cover = None
    if True:  # verbose:
        # find bound for cliques in solution
        max_intersect_num = graph.num_vertices**2 // 4
        if max_intersect_num < graph.num_edges:
            p = graph.n_choose_2(graph.num_vertices) - graph.num_edges
            t = int(np.sqrt(p))
            max_intersect_num = p + t if p > 0 else 1
        print("solution has at most {} cliques.".format(max_intersect_num))
    while the_cover is None:
        if True:  # verbose:
            print(
                "\ntesting for solutions with {}/{} cliques".format(
                    num_cliques, max_intersect_num
                )
            )
        the_cover = branch(graph, num_cliques, the_cover, iteration=0, iteration_max=3)
        num_cliques += 1

    return add_isolated_verts(the_cover)

def branch(graph, k_num_cliques, the_cover, iteration, iteration_max):
    """Helper function for `find_clique_min_cover()`.

    Describing the solution search space as a tree.
    This function tests whether the given node is a solution, and it branches if not.

    Parameters
    ----------
    graph : UndirectedDependenceGraph()
            Class for representing undirected graph and auxilliary data used in edge clique cover algorithm.

    k_num_cliques : int
                    Current depth of search; number of cliques in cover being testet for solution.

    the_cover : np.array
                Biadjacency matrix representing (possibly partial) edge clique cover.

    iteration: current iteration

    iteration_max: maximum number of iteration_max

    Returns
    -------
    2d numpy array or None
        Biadjacency matrix representing (complete) edge clique cover or None if cover is only partial.

    """

    iteration = iteration + 1
    branch_graph = graph.reducible_copy()
    # if the_cover is not None:
    #     print(the_cover)
    #     for clique in the_cover:  # this might not be necessary, since the_cover_prime is only +1 clique
    #         print('clique: {}'.format(clique))
    #         branch_graph.the_cover = [clique]
    #         branch_graph.cover_edges()  # only works one clique at a time, or on a list of edges
    branch_graph.the_cover = the_cover
    branch_graph.cover_edges()

    if branch_graph.num_edges == 0:
        return branch_graph.reconstruct_cover(the_cover)

    # branch_graph.the_cover = the_cover

    branch_graph.reduzieren(k_num_cliques)
    k_num_cliques = branch_graph.k_num_cliques

    if k_num_cliques < 0:
        return None

    if branch_graph.num_edges == 0:  # equiv to len(branch_graph.extant_edges_idx)==0
        return (
            branch_graph.the_cover
        )  # not in paper, but speeds it up slightly; or rather return None?

    chosen_nbrhood = branch_graph.choose_nbrhood()
    # print("num cliques: {}".format(len([x for x in max_cliques(chosen_nbrhood)])))
    for clique_nodes in max_cliques(chosen_nbrhood):
        if len(clique_nodes) == 1:  # then this vert has been rmed; quirk of max_cliques
            continue
        clique = np.zeros(branch_graph.unreduced.num_vertices, dtype=int)
        clique[clique_nodes] = 1
        union = (
            clique.reshape(1, -1)
            if branch_graph.the_cover is None
            else np.vstack((branch_graph.the_cover, clique))
        )

        # print(iteration)
        if iteration > iteration_max:
            return branch_graph.the_cover

        the_cover_prime = branch(
            branch_graph,
            k_num_cliques - 1,
            union,
            iteration,
            iteration_max=iteration_max,
        )
        if the_cover_prime is not None:
            return the_cover_prime
    return None

def max_cliques(nbrhood):
    """Adaptation of NetworkX code for finding all maximal cliques.

    Parameters
    ----------
    nbrhood : np.array
            Adjacency matrix for undirected (sub)graph.

    Returns
    -------
    generator
        set of all maximal cliques

    Notes
    -----
    Pieced together from nx.from_numpy_array and nx.find_cliques, which
    is output sensitive.

    """

    if len(nbrhood) == 0:
        return

    # convert adjacency matrix to nx style graph
    adj = {
        u: {v for v in np.nonzero(nbrhood[u])[0] if v != u} for u in range(len(nbrhood))
    }
    Q = [None]

    subg = set(range(len(nbrhood)))
    cand = set(range(len(nbrhood)))
    u = max(subg, key=lambda u: len(cand & adj[u]))
    ext_u = cand - adj[u]
    stack = []

    try:
        while True:
            if ext_u:
                q = ext_u.pop()
                cand.remove(q)
                Q[-1] = q
                adj_q = adj[q]
                subg_q = subg & adj_q
                if not subg_q:
                    yield Q[:]
                else:
                    cand_q = cand & adj_q
                    if cand_q:
                        stack.append((subg, cand, ext_u))
                        Q.append(None)
                        subg = subg_q
                        cand = cand_q
                        u = max(subg, key=lambda u: len(cand & adj[u]))
                        ext_u = cand - adj[u]
            else:
                Q.pop()
                subg, cand, ext_u = stack.pop()
    except IndexError:
        pass
    # note: max_cliques is a generator, so it's consumed after being
    # looped through once

def add_isolated_verts(cover):
    cover = cover.astype(bool)
    iso_vert_idx = np.flatnonzero(cover.sum(0) == 0)
    num_rows = len(iso_vert_idx)
    num_cols = cover.shape[1]
    iso_vert_cover = np.zeros((num_rows, num_cols), bool)
    iso_vert_cover[np.arange(num_rows), iso_vert_idx] = True
    return np.vstack((cover, iso_vert_cover))

class UndirectedDependenceGraph(object):
    r"""Adjacency matrix representation using a 2d numpy array.

    Upon initialization, this class is fairly standard implementation
    of an undirected graph. However, upon calling the :meth:`make_aux`
    method, an auxilliary data structure in the form of several new
    attributes is created, which are used by
    :meth:`medil.ecc_algorithms.find_clique_min_cover` according to
    the algorithm in :cite:`Gramm_2009`.

    Attributes
    ----------


    Notes
    -----
    The algorithms for finding the minMCM via ECC contain many
    algebraic operations, so adjacency matrix representation (via
    NumPy) is most covenient.

    The diag is used to store information when the graph is reduced,
    not to indicate self loops, so it is important that diag = 1 at
    init.

    """

    def __init__(self, adj_matrix, verbose=False):
        # doesn't behave well unless input is nparray
        self.adj_matrix = adj_matrix
        self.num_vertices = np.trace(adj_matrix)
        self.max_num_verts = len(adj_matrix)
        self.num_edges = np.triu(adj_matrix, 1).sum()
        self.verbose = verbose

    def add_edges(self, edges):
        v_1s = edges[:, 0]
        v_2s = edges[:, 1]
        self.adj_matrix[v_1s, v_2s] = 1
        self.adj_matrix[v_2s, v_1s] = 1
        self.num_edges = np.triu(self.adj_matrix, 1).sum()

    def rm_edges(self, edges):
        v_1s = edges[:, 0]
        v_2s = edges[:, 1]
        self.adj_matrix[v_1s, v_2s] = 0
        self.adj_matrix[v_2s, v_1s] = 0
        self.num_edges = np.triu(self.adj_matrix, 1).sum()

    def make_aux(self):
        # this makes the auxilliary structure described in INITIALIZATION in the paper

        # find neighbourhood for each vertex
        # each row corresponds to a unique edge
        max_num_edges = self.n_choose_2(self.max_num_verts)
        self.common_neighbors = np.zeros(
            (max_num_edges, self.max_num_verts), int
        )  # init

        # mapping of edges to unique row idx
        triu_idx = np.triu_indices(self.max_num_verts, 1)
        nghbrhd_idx = np.zeros((self.max_num_verts, self.max_num_verts), int)
        nghbrhd_idx[triu_idx] = np.arange(max_num_edges)
        # nghbrhd_idx += nghbrhd_idx.T
        self.get_idx = lambda edge: nghbrhd_idx[edge[0], edge[1]]

        # reverse mapping
        u, v = np.where(np.triu(np.ones_like(self.adj_matrix), 1))
        self.get_edge = lambda idx: (u[idx], v[idx])

        # compute actual neighborhood for each edge = (v_1, v_2)
        self.nbrs = lambda edge: np.logical_and(
            self.adj_matrix[edge[0]], self.adj_matrix[edge[1]]
        )

        extant_edges = np.transpose(np.triu(self.adj_matrix, 1).nonzero())
        self.extant_edges_idx = np.fromiter(
            {self.get_idx(edge) for edge in extant_edges}, dtype=int
        )
        extant_nbrs = np.array([self.nbrs(edge) for edge in extant_edges], int)
        extant_nbrs_idx = np.array([self.get_idx(edge) for edge in extant_edges], int)

        # from paper: set of N_{u, v} for all edges (u, v)
        self.common_neighbors[extant_nbrs_idx] = extant_nbrs

        # number of cliques for each node? assignments? if we set diag=0
        # num_cliques = common_neighbors.sum(0)

        # sum() of submatrix of graph containing exactly the rows/columns
        # corresponding to the nodes in common_neighbors(edge) using
        # logical indexing:

        # make mask to identify subgraph (closed common neighborhood of
        # nodes u, v in edge u,v)
        mask = lambda edge_idx: np.array(self.common_neighbors[edge_idx], dtype=bool)

        # make subgraph-adjacency matrix, and then subtract diag and
        # divide by two to get num edges in subgraph---same as sum() of
        # triu(subgraph-adjacency matrix) but probably a bit faster
        nbrhood = lambda edge_idx: self.adj_matrix[mask(edge_idx)][:, mask(edge_idx)]
        max_num_edges_in_nbrhood = (
            lambda edge_idx: (nbrhood(edge_idx).sum() - mask(edge_idx).sum()) // 2
        )

        # from paper: set of c_{u, v} for all edges (u, v)
        self.nbrhood_edge_counts = np.array(
            [
                max_num_edges_in_nbrhood(edge_idx)
                for edge_idx in np.arange(max_num_edges)
            ],
            int,
        )

        # important structs are:
        # self.common_neighbors
        # self.nbrhood_edge_counts
        # # and fun is
        # self.nbrs

    @staticmethod
    def n_choose_2(n):
        return n * (n - 1) // 2

    def reducible_copy(self):
        return ReducibleUndDepGraph(self)

    def convert_to_nde(self, name="temp"):
        with open(name + ".nde", "w") as f:
            f.write(str(self.max_num_verts) + "\n")
            for idx, node in enumerate(self.adj_matrix):
                f.write(str(idx) + " " + str(node.sum()) + "\n")
            for v1, v2 in np.argwhere(np.triu(self.adj_matrix)):
                f.write(str(v1) + " " + str(v2) + "\n")

class ReducibleUndDepGraph(UndirectedDependenceGraph):
    def __init__(self, udg):
        self.unreduced = udg.unreduced if hasattr(udg, "unreduced") else udg
        self.adj_matrix = udg.adj_matrix.copy()
        self.num_vertices = udg.num_vertices
        self.num_edges = udg.num_edges

        self.the_cover = None
        self.verbose = udg.verbose

        # from auxilliary structure if needed
        if not hasattr(udg, "get_idx"):
            udg.make_aux()
        self.get_idx = udg.get_idx

        # need to also update these all when self.cover_edges() is called? already done in rule_1
        self.common_neighbors = udg.common_neighbors.copy()
        self.nbrhood_edge_counts = udg.nbrhood_edge_counts.copy()

        # update when cover_edges() is called, actually maybe just extant_edges?
        self.extant_edges_idx = udg.extant_edges_idx.copy()
        self.nbrs = udg.nbrs
        self.get_edge = udg.get_edge

        if hasattr(udg, "reduced_away"):  # then rule_3 wasn't applied
            self.reduced_away = udg.reduced_away

    def reset(self):
        self.__init__(self.unreduced)

    def reduzieren(self, k_num_cliques):
        # reduce by first applying rule 1 and then repeatedly applying
        # rule 2 or rule 3 (both of which followed by rule 1 again)
        # until they don't apply
        if self.verbose:
            print("\t\treducing:")
        self.k_num_cliques = k_num_cliques
        self.reducing = True
        while self.reducing:
            self.reducing = False
            self.rule_1()
            self.rule_2()
            if self.k_num_cliques < 0:
                return
            if self.reducing:
                continue
            self.rule_3()

    def rule_1(self):
        # rule_1: Remove isolated vertices and vertices that are only
        # adjacent to covered edges

        isolated_verts = np.where(self.adj_matrix.sum(0) + self.adj_matrix.sum(1) == 2)[
            0
        ]
        if len(isolated_verts) > 0:  # then Rule 1 is applied
            if self.verbose:
                print("\t\t\tapplying Rule 1...")

            # update auxilliary attributes; LEMMA 2
            self.adj_matrix[isolated_verts, isolated_verts] = 0
            self.num_vertices -= len(isolated_verts)

            # decrease nbrhood edge counts
            for vert in isolated_verts:
                open_nbrhood = self.unreduced.adj_matrix[vert].copy()
                open_nbrhood[vert] = 0
                idx_nbrhoods_to_update = np.where(self.common_neighbors[:, vert] == 1)[
                    0
                ]
                tiled = np.tile(
                    open_nbrhood, (len(idx_nbrhoods_to_update), 1)
                )  # instead of another loop
                to_subtract = np.logical_and(
                    tiled, self.common_neighbors[idx_nbrhoods_to_update]
                ).sum(1)
                self.nbrhood_edge_counts[idx_nbrhoods_to_update] -= to_subtract
                # my own addition:
                # self.nbrhood[:, vert] = 0

            # remove isolated_verts from common neighborhoods
            self.common_neighbors[:, isolated_verts] = 0

        # max_num_edges = self.n_choose_2(self.unreduced.max_num_verts)
        # mask = lambda edge_idx: np.array(self.common_neighbors[edge_idx], dtype=bool)

        # # make subgraph-adjacency matrix, and then subtract diag and
        # # divide by two to get num edges in subgraph---same as sum() of
        # # triu(subgraph-adjacency matrix) but probably a bit faster
        # nbrhood = lambda edge_idx: self.adj_matrix[mask(edge_idx)][:, mask(edge_idx)]
        # max_num_edges_in_nbrhood = lambda edge_idx: (nbrhood(edge_idx).sum() - mask(edge_idx).sum()) // 2

        # # from paper: set of c_{u, v} for all edges (u, v)
        # self.nbrhood_edge_counts = np.array([max_num_edges_in_nbrhood(edge_idx) for edge_idx in np.arange(max_num_edges)], int)
        # # assert (nbrhood_edge_counts==self.nbrhood_edge_counts).all()
        # # print(nbrhood_edge_counts, self.nbrhood_edge_counts)
        # # need to fix!!!!!!!! update isn't working; so just recomputing for now
        # # # # # # # # but actually update produces correct result though recomputing doesn't?

    def rule_2(self):
        # rule_2: If an uncovered edge {u,v} is contained in exactly
        # one maximal clique C, i.e., the common neighbors of u and v
        # induce a clique, then add C to the solution, mark its edges
        # as covered, and decrease k by one

        score = self.n_choose_2(self.common_neighbors.sum(1)) - self.nbrhood_edge_counts
        # score of n implies edge is in exactly n+1 maximal cliques,
        # so we want edges with score 0

        clique_idxs = np.where(score[self.extant_edges_idx] == 0)[0]

        if clique_idxs.size > 0:
            clique_idx = clique_idxs[-1]
            if self.verbose:
                print("\t\t\tapplying Rule 2...")
            clique = self.common_neighbors[self.extant_edges_idx[clique_idx]].copy()
            self.the_cover = (
                clique.reshape(1, -1)
                if self.the_cover is None
                else np.vstack((self.the_cover, clique))
            )
            self.cover_edges()
            self.k_num_cliques -= 1
            self.reducing = True
        # start the loop over so Rule 1 can 'clean up'
        # self.common_neighbors[clique_idxs[0]] = 0  # zero out row, to update struct? not in paper?

    def rule_3(self):
        # rule_3: Consider a vertex v that has at least one
        # prisoner. If each prisoner is connected to at least one
        # vertex other than v via an uncovered edge (automatically
        # given if instance is reduced w.r.t. rules 1 and 2), and the
        # prisoners dominate the exit,s then delete v. To reconstruct
        # a solution for the unreduced instance, add v to every clique
        # containing a prisoner of v.

        for vert, nbrhood in enumerate(self.adj_matrix):
            if nbrhood[vert] == 0:  # then nbrhood is empty
                continue
            exits = np.zeros(self.unreduced.max_num_verts, bool)
            nbrs = np.flatnonzero(nbrhood)
            for nbr in nbrs:
                if (nbrhood.astype(int) - self.adj_matrix[nbr].astype(int) == -1).any():
                    exits[nbr] = True
            # exits[j] == True iff j is an exit for vert
            # prisoners[j] == True iff j is a prisoner of vert
            prisoners = np.logical_and(~exits, self.adj_matrix[vert])
            prisoners[vert] = False  # a vert isn't its own prisoner

            # check if each exit is adjacent to at least one prisoner
            prisoners_dominate_exits = self.adj_matrix[prisoners][:, exits].sum(0)
            if prisoners_dominate_exits.all():  # apply the rule
                self.reducing = True
                if self.verbose:
                    print("\t\t\tapplying Rule 3...")

                # mark edges covered so rule_1 will delete vertex
                edges = np.array(
                    [
                        [vert, u]
                        for u in np.flatnonzero(self.adj_matrix[vert])
                        if vert != u
                    ]
                )
                edges.sort()  # so they're in triu, so that self.get_idx works
                self.cover_edges(edges)

                # keep track of deleted nodes and their prisoners for reconstructing solution
                if not hasattr(self, "reduced_away"):
                    self.reduced_away = np.zeros_like(self.adj_matrix, bool)
                self.reduced_away[vert] = prisoners
                break

    def choose_nbrhood(self):
        # only compute for existing edges
        common_neighbors = self.common_neighbors[self.extant_edges_idx]
        nbrhood_edge_counts = self.nbrhood_edge_counts[self.extant_edges_idx]
        score = self.n_choose_2(common_neighbors.sum(1)) - nbrhood_edge_counts

        # idx in reduced idx list, not from full edge list
        chosen_edge_idx = score.argmin()

        mask = common_neighbors[chosen_edge_idx].astype(bool)

        chosen_nbrhood = self.adj_matrix.copy()
        chosen_nbrhood[~mask, :] = 0
        chosen_nbrhood[:, ~mask] = 0
        if chosen_nbrhood.sum() <= mask.sum():
            raise ValueError("This nbrhood has no edges")
        return chosen_nbrhood

    def cover_edges(self, edges=None):
        if edges is None:
            # always call after updating the cover; only on single, recently added clique
            if self.the_cover is None:
                return  # self.adj_matrix

            # change edges to 0 if they're covered
            clique = self.the_cover[-1]
            covered = np.where(clique)[0]

            # trick for getting combinations from idx
            comb_idx = np.triu_indices(len(covered), 1)

            # actual pairwise combinations; ie all edges (v_i, v_j) covered by the clique
            covered_edges = np.empty((len(comb_idx[0]), 2), int)
            covered_edges[:, 0] = covered[comb_idx[0]]
            covered_edges[:, 1] = covered[comb_idx[1]]

        else:
            covered_edges = edges

        # cover (remove from reduced_graph) edges
        self.rm_edges(covered_edges)
        # update extant_edges_idx
        rmed_edges_idx = [self.get_idx(edge) for edge in covered_edges]
        extant_rmed_edges_idx = [
            edge for edge in rmed_edges_idx if edge in self.extant_edges_idx
        ]
        idx_idx = np.array(
            [np.where(self.extant_edges_idx == idx) for idx in extant_rmed_edges_idx],
            int,
        ).flatten()

        self.extant_edges_idx = np.delete(self.extant_edges_idx, idx_idx)
        # now here do all the updates to nbrs?----actually probably don't want this? see 2clique house example

        # update self.common_neighbors
        # self.common_neighbors[rmed_edges_idx] = 0   # zero out rows covered edges; maybe not necessary, since common_neighbors is (probably?) only called with extant_edges_idx?

        if self.verbose:
            print("\t\t\t{} uncovered edges remaining".format(self.num_edges))

        # if edges is None:
        #     cover_orig = self.the_cover
        #     while self.the_cover.shape[0] > 1:
        #         self.the_cover = self.the_cover[:-1]
        #         self.cover_edges()
        #     self.the_cover = cover_orig

    def reconstruct_cover(self, the_cover):
        if not hasattr(self, "reduced_away"):  # then rule_3 wasn't applied
            return the_cover

        # add the reduced away vert to all covering cliques containing at least one of its prisoners
        to_expand = np.flatnonzero(self.reduced_away.sum(1))
        for vert in to_expand:
            prisoners = self.reduced_away[vert]
            tiled_prisoners = np.tile(
                prisoners, (len(the_cover), 1)
            )  # instead of another loop
            cliques_to_update_mask = (
                np.logical_and(tiled_prisoners, the_cover).sum(1).astype(bool)
            )
            the_cover[cliques_to_update_mask, vert] = 1

        return the_cover

if __name__ == "__main__":
    idx = int(snakemake.wildcards.idx)
    sparsity = float(snakemake.params.get("sparsity", 0.5))
    n_latent = int(snakemake.params.get("n_latent", 3))
    n_observed = int(snakemake.params.get("n_observed", 5))
    seed = int(snakemake.params.get("seed", 42))

# select one graph from the benchmark graphs or generate a random one
if idx in biadj_dict:
    biadj_matrix = biadj_dict[idx]
else:
    rng = default_rng(seed)
    biadj_matrix = biadj(
        num_meas=n_observed,
        density=1 - sparsity,  
        one_pure_child=True,
        num_latent=n_latent,
        rng=rng
    )

# output
np.savetxt(snakemake.output.biadj, biadj_matrix, delimiter=",", fmt="%1u")
