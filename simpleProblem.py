from numbers import Integral
import numpy as np
from pymoo.core.problem import Problem


class Knapsack(Problem):

    def __init__(self, values_1, values_2, weights, capacities):
        super().__init__(n_var=len(values_1),
                         n_obj=2,
                         n_constr=len(capacities),
                         xl=0,
                         xu=len(capacities))
        self.values_1 = values_1
        self.values_2 = values_2
        self.weights = weights
        self.capacities = capacities

    def _evaluate(self, X, out, *args, **kwargs):
        # Calculate the value and weight of each item for each knapsack
        knapsack_num = self.capacities.shape[0]
        values_1 = np.zeros(X.shape[0])
        values_2 = np.zeros(X.shape[0])
        weights = np.zeros((X.shape[0], knapsack_num))

        # calculate the value and weight of each item
        for i in range(knapsack_num):
            values_1 += np.sum((X == i+1) * self.values_1, axis=1)
            values_2 += np.sum((X == i+1) * self.values_2, axis=1)
            weights[:, i] = np.sum((X == i+1) * self.weights, axis=1)

        # calculate the total value and weight of the knapsack
        out["F"] = np.column_stack([1/values_1, 1/values_2])
        out["G"] = weights - self.capacities
    
    def evaluate_x(self, X):
        values_1 = np.zeros(X.shape[0])
        values_2 = np.zeros(X.shape[0])

        for i in range(self.capacities.shape[0]):
            values_1 += np.sum((X == i+1) * self.values_1, axis=1)
            values_2 += np.sum((X == i+1) * self.values_2, axis=1)

        return np.column_stack([-values_1, values_2])


class Knapsack_one(Problem):

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
    
    def evaluate_x(self, X):
        values = np.sum(X * self.values, axis=1)
        volume = np.sum(X * self.volume, axis=1)
        return np.column_stack([-values, volume])