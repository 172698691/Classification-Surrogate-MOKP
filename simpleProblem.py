import numpy as np
from pymoo.core.problem import Problem


class Knapsack(Problem):

    def __init__(self, values, volume, weights, capacity):
        super().__init__(n_var=len(values),
                         n_obj=2,
                         n_constr=1,
                         xl=0,
                         xu=1)
        self.values = values
        self.weights = weights
        self.volume = volume
        self.capacity = capacity

    def _evaluate(self, X, out, *args, **kwargs):
        # calculate the value and weight of each item
        values = np.sum(X * self.values, axis=1)
        volume = np.sum(X * self.volume, axis=1)
        weights = np.sum(X * self.weights, axis=1)

        # calculate the total value and weight of the knapsack
        out["F"] = np.column_stack([-values, volume])
        out["G"] = weights - self.capacity