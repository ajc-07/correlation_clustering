import sys
from itertools import combinations, product, chain
from collections import defaultdict, Counter
import random
import networkx as nx
import numpy as np
from scipy.stats import spearmanr
from networkx.algorithms.dag import transitive_closure
import six
sys.modules['sklearn.externals.six'] = six
import mlrose_hiive as mlrose
import time
from scipy.optimize import linear_sum_assignment
import multiprocessing as mp


def cluster_correlation_search(G, k, max_attempts=200, max_iters=5000, initial=None, split_flag=True):
    """
    Apply correlation clustering with exactly k clusters. Assumes that negative edges have weights < 0,
    and positive edges have weights >= 0, that edges with nan have been removed, and that weights are
    stored under edge attribute G[i][j]['weight'].

    :param G: graph
    :param k: exact number of clusters to find  # CHANGED: replaced 's' parameter with 'k'
    :param max_attempts: number of restarts for optimization
    :param max_iters: number of iterations for optimization
    :param initial: optional clustering for initialization
    :param split_flag: optional flag, if non-evidence cluster should be split
    :return classes, stats: list of clusters, stats dict
    """
    start_time = time.time()
    stats = {}
    G = G.copy()

    if initial is None or initial == []:
        classes = cluster_connected_components(G)
    else:
        classes = initial

    # Mapping nodes to indices and back
    n2i = {node: i for i, node in enumerate(G.nodes())}
    i2n = {i: node for node, i in n2i.items()}
    n2c = {n2i[node]: ci for ci, cluster in enumerate(classes) for node in cluster}

    # Separate positive and negative edges
    edges_positive = set(
        (n2i[i], n2i[j], G[i][j]['weight'])
        for (i, j) in G.edges() if G[i][j]['weight'] >= 0.0
    )
    edges_negative = set(
        (n2i[i], n2i[j], G[i][j]['weight'])
        for (i, j) in G.edges() if G[i][j]['weight'] < 0.0
    )

    Linear_loss = Loss(
        'linear_loss', edges_positive=edges_positive, edges_negative=edges_negative, k_exact = k
    )

    # Define initial state array
    init_state = np.array([n2c.get(n, 0) for n in sorted(n2c.keys())])
    loss_init = Linear_loss.loss(init_state)

    if loss_init == 0.0:
        classes.sort(key=lambda x: -len(x))
        end_time = time.time()
        stats = {
            'k': k,                      # CHANGED: report 'k' instead of 's'
            'max_attempts': max_attempts,
            'max_iters': max_iters,
            'split_flag': split_flag,
            'runtime': (end_time - start_time)/60,
            'loss': loss_init
        }
        return classes, stats

    # Initialize loss-to-states mapping
    l2s = defaultdict(list)
    l2s[loss_init].append((init_state, len(classes)))

    # Initialize multiprocessing pool
    pool = mp.Pool(mp.cpu_count())

    # CHANGED: Only run simulated annealing once for n = k
    solutions = pool.starmap(
        Linear_loss.optimize_simulated_annealing,
        [(k, classes, G.nodes(), init_state, max_attempts, max_iters)]
    )
    pool.close()

    # Merge solutions from pool
    for sol in solutions:
        for loss_val, states in sol.items():
            l2s[loss_val].extend(states)

    # Select the best state
    best_loss = min(l2s.keys())
    chosen_states = l2s[best_loss]
    best_state, _ = random.choice(chosen_states)

    # Map state labels back to node sets
    c2n = defaultdict(list)
    for idx_label, comp_label in enumerate(best_state):
        c2n[comp_label].append(i2n[idx_label])
    classes = [set(nodes) for nodes in c2n.values()]

    if split_flag:
        classes = split_non_evidence_clusters(G, classes)

    classes.sort(key=lambda x: -len(x))
    end_time = time.time()

    stats = {
        'k': k,                        # CHANGED: report 'k' instead of 's'
        'max_attempts': max_attempts,
        'max_iters': max_iters,
        'split_flag': split_flag,
        'runtime': (end_time - start_time)/60,
        'loss': best_loss
    }

    return classes, stats


class Loss(object):
    """
    Wrapper for different loss computations.
    """
    def __init__(self, fitness_fn, edges_positive=None, edges_negative=None,
                 edges_min=None, edges_max=None, signs=None, k_exact=None, penalty=1e9):
        # Still wasn't doing exact number of clusters, so add huge loss if not exactly k
        self.k_exact = k_exact
        self.penalty = penalty
        self.edges_positive = edges_positive
        self.edges_negative = edges_negative
        self.edges_min = edges_min
        self.edges_max = edges_max
        self.signs = signs
        if fitness_fn == 'test_loss':
            self.fitness_fn = self.test_loss
        if fitness_fn == 'linear_loss':
            self.fitness_fn = self.linear_loss
        if fitness_fn == 'binary_loss':
            self.fitness_fn = self.binary_loss
        if fitness_fn == 'binary_loss_poles':
            self.fitness_fn = self.binary_loss_poles

    def loss(self, state):
        return self.fitness_fn(state)

    def test_loss(self, state):
        return 50.0

    def linear_loss(self, state):
        # original cost
        cost = (
            np.sum([w             for (i,j,w) in self.edges_positive if state[i]!=state[j]])
          + np.sum([abs(w)        for (i,j,w) in self.edges_negative if state[i]==state[j]])
        )

        # If there exists a k_exact, then if number of clusters isn't exact then add huge loss
        if self.k_exact is not None:
            used = len(set(state))
            if used != self.k_exact: # Wrong count -> huge penalty
                cost += self.penalty * abs(used - self.k_exact)

        return cost

    def binary_loss(self, state):
        loss_pos = len([1 for (i,j,w) in self.edges_positive if state[i]!=state[j]])
        loss_neg = len([1 for (i,j,w) in self.edges_negative if state[i]==state[j]])
        if self.signs==['pos','neg']:
            return loss_pos + loss_neg
        if self.signs==['pos']:
            return loss_pos
        if self.signs==['neg']:
            return loss_neg
        return float('nan')

    def binary_loss_poles(self, state):
        loss_min = len([1 for (i,j) in self.edges_min if state[i]==state[j]])
        loss_max = len([1 for (i,j) in self.edges_max if state[i]!=state[j]])
        if self.signs==['min','max']:
            return loss_min + loss_max
        if self.signs==['min']:
            return loss_min
        if self.signs==['max']:
            return loss_max
        return float('nan')

    def optimize_simulated_annealing(self, n, classes, nodes, init_state, max_attempts, max_iters):
        # CHANGED: use n (==k) as the fixed max_val for clustering labels
        np.random.seed()
        fitness_fn = mlrose.CustomFitness(self.fitness_fn)
        l2s_ = defaultdict(list)

        # With initial state
        problem = mlrose.DiscreteOpt(
            length=len(nodes), fitness_fn=fitness_fn,
            maximize=False, max_val=n  # CHANGED: force exactly k clusters
        )
        schedule = mlrose.GeomDecay()
        best_state, best_fitness, _ = mlrose.simulated_annealing(
            problem, schedule=schedule, init_state=init_state,
            max_attempts=max_attempts, max_iters=max_iters, n_restarts=10
        )
        l2s_[best_fitness].append((best_state, n))

        np.random.seed()
        # Repeat without initial state
        problem = mlrose.DiscreteOpt(
            length=len(nodes), fitness_fn=fitness_fn,
            maximize=False, max_val=n  # CHANGED: force exactly k clusters
        )
        schedule = mlrose.GeomDecay()
        best_state2, best_fitness2, _ = mlrose.simulated_annealing(
            problem, schedule=schedule,
            max_attempts=max_attempts, max_iters=max_iters, n_restarts=10
        )
        l2s_[best_fitness2].append((best_state2, n))

        return dict(l2s_)


def cluster_connected_components(G, is_non_value=lambda x: np.isnan(x)):
    """
    Apply connected component clustering on positive edges.
    """
    G = G.copy()
    edges_negative = [
        (i,j) for (i,j) in G.edges()
        if G[i][j]['weight'] < 0.0 or is_non_value(G[i][j]['weight'])
    ]
    G.remove_edges_from(edges_negative)
    components = nx.connected_components(G)
    classes = [set(c) for c in components]
    classes.sort(key=lambda x: list(x)[0])
    return classes


def split_non_evidence_clusters(G, clusters, is_non_value=lambda x: np.isnan(x)):
    """
    Split clusters by removing negative edges and reconnected components.
    """
    G = G.copy()
    nodes_in = [node for c in clusters for node in c]
    edges_negative = [
        (i,j) for (i,j) in G.edges()
        if G[i][j]['weight'] < 0.0 or is_non_value(G[i][j]['weight'])
    ]
    G.remove_edges_from(edges_negative)
    classes_out = []
    for cluster in clusters:
        sub = G.subgraph(cluster)
        for comp in nx.connected_components(sub):
            classes_out.append(set(comp))
    nodes_out = [n for c in classes_out for n in c]
    if set(nodes_in) != set(nodes_out) or len(nodes_in) != len(nodes_out):
        sys.exit('Breaking: node mismatch after split.')
    return classes_out
