import numpy as np


def get_non_dominated_solutions(solutions):
    def is_dominated(x, y):
        return all(x[i] <= y[i] for i in range(len(x))) and any(x[i] < y[i] for i in range(len(x)))

    non_dominated = []
    for i, x in enumerate(solutions):
        dominated = False
        for j, y in enumerate(solutions):
            if i != j and is_dominated(y, x):
                dominated = True
                break
        if not dominated:
            non_dominated.append(x)
    return np.array(non_dominated)