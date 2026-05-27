# cear_pilot/analysis/phi/information.py
# ---------------------------------------------------------------------------
# Vendored from: https://github.com/pigozzif/PhiRL/blob/master/information.py
# Original authors: Pigozzi & Levin (2026). Used here for direct method
# comparability with their paper.
#
# Local change: lattice pickle is loaded from THIS DIRECTORY (not cwd),
# so imports work regardless of where Python is launched from.
# ---------------------------------------------------------------------------

import os
import pickle
from copy import deepcopy

import networkx as nx
import numpy as np
from scipy.stats import linregress, zscore, pearsonr, multivariate_normal, norm
from scipy.stats import t as student_t

_HERE = os.path.dirname(os.path.abspath(__file__))
_LATTICE_PATH = os.path.join(_HERE, "phi_lattice_22.pickle")

if not os.path.exists(_LATTICE_PATH):
    raise FileNotFoundError(
        f"Missing {_LATTICE_PATH}. Download it from "
        "https://github.com/pigozzif/PhiRL/blob/master/phi_lattice_22.pickle "
        "and place it next to this file."
    )

LATTICE_ORIG = pickle.load(open(_LATTICE_PATH, "rb"))

DISTANCES = nx.shortest_path_length(LATTICE_ORIG, target=(((0,), (1,)), ((0,), (1,))))
ORDER = []
for distance in range(max(DISTANCES.values()) + 1):
    ORDER += [key for key in DISTANCES.keys() if DISTANCES[key] == distance]

PHIR_ATOMS = {  # Only used for the phi_r function.
    (((0,), (1,)), ((0, 1),)),
    (((1,),), ((0, 1),)),
    (((0, 1),), ((0,),)),
    (((0, 1),), ((0,), (1,))),
    (((0, 1),), ((1,),)),
    (((0, 1),), ((0, 1),)),
    (((0,),), ((1,),)),
    (((1,),), ((0,),)),
}


def corrected_zscore(data, axis=1, noise=10 ** -6):
    stds = data.std(axis=1, keepdims=True)
    dead_mask = stds.squeeze() < 1e-6
    data[dead_mask, :] = np.random.randn(dead_mask.sum(), data.shape[1]) * noise
    data = zscore(data, axis=axis)
    return data


def local_entropy_1d(idx1, x):
    mu = x[idx1].mean()
    sigma = x[idx1].std()
    entropy = -np.log(norm.pdf(x[idx1], loc=mu, scale=sigma))
    return entropy


def local_entropy_nd(x, eps=1e-6):
    if x.shape[0] == 1:
        return local_entropy_1d(0, x)
    else:
        cov = np.cov(x, ddof=0)
        eps_matrix = eps * np.trace(cov) / cov.shape[0]
        cov += np.eye(cov.shape[0]) * eps_matrix
        means = x.mean(axis=-1)
        entropy = -np.log(multivariate_normal.pdf(x.T, mean=means, cov=cov))
        return entropy


def local_phi_min(idx1, idx2, atom, x, lag=1):
    n1 = x.shape[1]
    edge = x[[idx1, idx2], :]
    i_plus = np.repeat(np.inf, n1 - lag)
    i_minus = np.repeat(np.inf, n1 - lag)
    len_atom_0 = len(atom[0])
    len_atom_1 = len(atom[1])
    for i in range(len_atom_0):
        edge_i = edge[((atom[0][i]),)][:, :-lag]
        h_edge_i = local_entropy_nd(edge_i)
        i_plus = np.minimum(i_plus, h_edge_i)
        for j in range(len_atom_1):
            joint = np.squeeze(
                np.vstack((
                    edge[(atom[0][i],)][:, :-lag],
                    edge[(atom[1][j],)][:, lag:]
                ))
            )
            marginal = edge[(atom[1][j],)][:, lag:]
            conditional = np.subtract(local_entropy_nd(joint), local_entropy_nd(marginal))
            i_minus = np.minimum(i_minus, conditional)
    return np.subtract(i_plus, i_minus)


def local_phi_id(idx1, idx2, x):
    lattice = deepcopy(LATTICE_ORIG)
    for atom in ORDER:
        lattice.nodes[atom]["phi_min"] = local_phi_min(idx1, idx2, atom, x)
        if atom == (((0,), (1,)), ((0,), (1,))):
            lattice.nodes[atom]["pi"] = lattice.nodes[atom]["phi_min"]
        else:
            lattice.nodes[atom]["pi"] = np.subtract(
                lattice.nodes[atom]["phi_min"],
                np.vstack(([lattice.nodes[a]["pi"] for a in lattice.nodes[atom]["descendants"]])).sum(axis=0)
            )
    return lattice


def local_phi_r(phi_lattice):
    phir = phi_lattice.nodes[(((0,),), ((0, 1),))]["pi"]
    for atom in PHIR_ATOMS:
        phir += phi_lattice.nodes[atom]["pi"]
    return phir


def mutual_information_matrix_fast(x, alpha=0.05, lag=0, bonferonni=True):
    n0, t = x.shape
    n_tests = (n0 ** 2 - n0) / 2
    alpha_corr = alpha / n_tests if bonferonni else alpha

    if lag == 0:
        r = np.corrcoef(x)
    else:
        x_f = x[:, :-lag]
        x_b = x[:, lag:]
        r1 = np.corrcoef(np.concatenate([x_f, x_b], axis=0))
        r2 = np.corrcoef(np.concatenate([x_b, x_f], axis=0))
        r1 = r1[:n0, n0:]
        r2 = r2[:n0, n0:]
        r = (r1 + r2) / 2

    df = t - 2
    r = np.clip(r, -0.999999, 0.999999)
    t_stat = r * np.sqrt(df / (1 - r ** 2))
    pvals = 2 * (1 - student_t.cdf(np.abs(t_stat), df))
    sig_mask = pvals < alpha_corr

    mi = np.zeros_like(r)
    mi[sig_mask] = -0.5 * np.log(1 - r[sig_mask] ** 2)
    np.fill_diagonal(mi, 0.0)
    return mi


def minimum_information_bipartition(mi_mat, noise=False, noise_level=10 ** -6):
    n0 = mi_mat.shape[0]
    if noise:
        mi_mat_corr = np.add(mi_mat, noise_level)
    else:
        mi_mat_corr = 1 * mi_mat
    g = nx.from_numpy_array(mi_mat_corr, create_using=nx.Graph())
    fiedler = nx.fiedler_vector(g, weight="weight", normalized=False)
    bipartition = [
        [i for i in range(n0) if fiedler[i] > 0],
        [i for i in range(n0) if fiedler[i] < 0]
    ]
    return bipartition
