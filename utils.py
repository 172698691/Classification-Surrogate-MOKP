import numpy as np
from pymoo.core.indicator import Indicator
from pymoo.core.mutation import Mutation
from scipy.spatial.distance import cdist


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
