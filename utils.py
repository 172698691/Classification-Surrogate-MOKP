import numpy as np


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
