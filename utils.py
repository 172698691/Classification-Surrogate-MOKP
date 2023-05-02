import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import ranksums
from pymoo.core.indicator import Indicator
from pymoo.core.mutation import Mutation
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize


class PD(Indicator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do(self, pop_obj, **kwargs):
        pop_length = pop_obj.shape[0]
        connection_matrix = np.eye(pop_length, dtype=bool)
        distance_matrix = cdist(pop_obj, pop_obj, metric='minkowski', p=0.1)
        np.fill_diagonal(distance_matrix, np.inf)
        score = 0

        for k in range(pop_length - 1):
            while True:
                min_distances = distance_matrix.min(axis=1)
                min_indices = distance_matrix.argmin(axis=1)
                max_index = min_distances.argmax()
                if distance_matrix[min_indices[max_index], max_index] != -np.inf:
                    distance_matrix[min_indices[max_index], max_index] = np.inf
                if distance_matrix[max_index, min_indices[max_index]] != -np.inf:
                    distance_matrix[max_index, min_indices[max_index]] = np.inf
                connected_nodes = connection_matrix[max_index, :]
                while not connected_nodes[min_indices[max_index]]:
                    new_connected_nodes = np.any(connection_matrix[connected_nodes, :], axis=0)
                    if np.array_equal(connected_nodes, new_connected_nodes):
                        break
                    else:
                        connected_nodes = new_connected_nodes
                if not connected_nodes[min_indices[max_index]]:
                    break

            connection_matrix[max_index, min_indices[max_index]] = True
            connection_matrix[min_indices[max_index], max_index] = True
            distance_matrix[max_index, :] = -np.inf
            score += min_distances[max_index]

        return score


class IntegerRandomMutation(Mutation):
    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        X = np.copy(X)
        n_var = X.shape[1]

        for i in range(X.shape[0]):
            for j in range(n_var):
                if np.random.random() < self.prob:
                    X[i, j] = np.random.randint(problem.xl[j], problem.xu[j] + 1)

        return X


def get_non_dominated_solutions(solutions, return_index=False):
    def is_dominated(x, y):
        return all(x[i] <= y[i] for i in range(len(x))) and any(x[i] < y[i] for i in range(len(x)))

    non_dominated = []
    non_dominated_indices = []
    for i, x in enumerate(solutions):
        dominated = False
        for j, y in enumerate(solutions):
            if i != j and is_dominated(y, x):
                dominated = True
                break
        if not dominated:
            non_dominated.append(x)
            non_dominated_indices.append(i)
    if return_index:
        return np.array(non_dominated), np.array(non_dominated_indices)
    else:
        return np.array(non_dominated)


def cal_igd(problem, nsga_hist_F, surrogate_hist_F):
    # get the parato front of the problem
    algorithm = NSGA2(
        pop_size=300,
        sampling=BinaryRandomSampling(),
        crossover=PointCrossover(n_points=2),
        mutation=BitflipMutation(),
        eliminate_duplicates=True
    )
    res = minimize(problem,
                algorithm,
                get_termination("n_gen", 600))
    parato_front = res.F
    # calculate the igd
    metric = IGD(parato_front, zero_to_one=True)
    nsga_y = [metric.do(_F) for _F in nsga_hist_F]
    surrogate_y = [metric.do(_F) for _F in surrogate_hist_F]
    return nsga_y, surrogate_y


def cal_hv(res_F_list, hist_F_list):
    approx_ideal_list, approx_nadir_list = [], []
    for F in res_F_list:
        approx_ideal, approx_nadir = F.min(axis=0), F.max(axis=0)
        approx_ideal_list.append(approx_ideal)
        approx_nadir_list.append(approx_nadir)
    
    approx_ideal = np.min(approx_ideal_list, axis=0)
    approx_nadir = np.max(approx_nadir_list, axis=0)
    metric = Hypervolume(ref_point= np.array([1.1, 1.1]),
                norm_ref_point=False,
                zero_to_one=True,
                ideal=approx_ideal,
                nadir=approx_nadir)
    
    y_list = []
    for hist_F in hist_F_list:
        y = [metric.do(_F) for _F in hist_F]
        y_list.append(y)
    
    return y_list


def cal_spacing(res_list, hist_F_list):
    approx_ideal_list, approx_nadir_list = [], []
    for res in res_list:
        F = res.opt.get("F")
        approx_ideal, approx_nadir = F.min(axis=0), F.max(axis=0)
        approx_ideal_list.append(approx_ideal)
        approx_nadir_list.append(approx_nadir)
    
    approx_ideal = np.min(approx_ideal_list, axis=0)
    approx_nadir = np.max(approx_nadir_list, axis=0)
    metric = SpacingIndicator(zero_to_one=True,
                ideal=approx_ideal,
                nadir=approx_nadir)
    
    y_list = []
    for hist_F in hist_F_list:
        y = [metric.do(_F) for _F in hist_F]
        y_list.append(y)
    
    return y_list


def cal_pd(hist_F_list):
    metric = PD()

    y_list = []
    for hist_F in hist_F_list:
        y = [metric(_F) for _F in hist_F]
        y_list.append(y)
    
    return y_list


def get_mean_every_n(x_list, y_list, n):
    x_mean, y_mean = [], []
    x_mean = x_list[::n]
    y_mean = np.array([np.mean(y_list[i:i+n]) for i in range(0, len(y_list), n)])
    return x_mean, y_mean


def rank_sum_test(a1, a2, sig_level=0.05):
    # Perform the Wilcoxon Rank Sum Test
    _, p_value_l = ranksums(a1, a2, alternative='less')
    _, p_value_g = ranksums(a1, a2, alternative='greater')
    # Check if the p-value is less than the significance level
    if p_value_l < sig_level:
        print("Better.")
    elif p_value_g < sig_level:
        print("Worse.")
    else:
        print("Similar.")
