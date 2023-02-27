import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from pymoo.core.problem import Problem
from pymoo.core.survival import Survival
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.indicators.igd import IGD

from simpleProblem import Knapsack
from surrogate import SurrogateNSGA2


def main():
    # define the problem
    n_items = 300
    values = np.random.randint(1, 50, size=n_items)
    volume = np.random.randint(1, 30, size=n_items)
    weights = np.random.randint(1, 20, size=n_items)
    capacity = 0.6*np.sum(weights)
    problem = Knapsack(values, volume, weights, capacity)

    # define the surrogate model
    classifier_name = "GB"
    classifier_arg={'n_estimators': 200, 'learning_rate': 0.15, 'max_depth': 5}

    # get the parato front of the problem
    algorithm = NSGA2(
        pop_size=300,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True
    )
    res = minimize(problem,
                algorithm,
                get_termination("n_gen", 600))
    parato_front = res.F
    metric = IGD(parato_front, zero_to_one=True)

    # define the algorithm
    algorithm = NSGA2(
        pop_size=100,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True
    )
    surrogate_algorithm = SurrogateNSGA2(
        pop_size=100,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
        classifier_name=classifier_name,
        classifier_arg=classifier_arg
    )

    # define the termination criterion
    termination = get_termination("n_gen", 50)

    # define a list to store all objective values
    igd, surrogate_igd = [], []

    # define the callback function
    def callback(algorithm):
        igd.append(metric.do(algorithm.pop.get("F")))
    def sur_callback(algorithm):
        surrogate_igd.append(metric.do(algorithm.pop.get("F")))

    # run the optimization
    nsga_res = minimize(problem,
                algorithm,
                termination,
                callback=callback)
    surrogate_res = minimize(problem,
                surrogate_algorithm,
                termination,
                callback=sur_callback)

    # # print the results
    print("Best solution found:")
    # # print("X = ", res.X.astype(int))
    # print("F = ", res.F)
    # print("SF = ", surrogate_res.F)

    # plot res.F
    # plt.scatter(surrogate_res.F[:, 0], surrogate_res.F[:, 1], color='r')
    # plt.scatter(res.F[:, 0], res.F[:, 1], color='g')
    # plt.show()

    # plot igd
    plt.plot(igd, color='g', label='NSGA2')
    plt.plot(surrogate_igd, color='r', label='Surrogate')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
