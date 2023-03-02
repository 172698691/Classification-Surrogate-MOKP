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
from pymoo.indicators.hv import Hypervolume

from simpleProblem import Knapsack
from surrogate import SurrogateNSGA2


def main():
    # set config
    criterion = 'hv'

    # define the problem
    n_items = 300
    values = np.random.randint(1, 50, size=n_items)
    volume = np.random.randint(1, 30, size=n_items)
    weights = np.random.randint(1, 20, size=n_items)
    capacity = 0.6*np.sum(weights)
    problem = Knapsack(values, volume, weights, capacity)

    # define the surrogate model
    # classifier_name = "GB"
    # classifier_arg={'n_estimators': 200, 'learning_rate': 0.15, 'max_depth': 5}
    classifier_name = "RF"
    classifier_arg={'n_estimators': 150}

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
    termination = get_termination("n_eval", 10000)

    # run the optimization
    nsga_res = minimize(problem,
                algorithm,
                termination,
                save_history=True)
    surrogate_res = minimize(problem,
                surrogate_algorithm,
                termination,
                save_history=True)
    
    nsga_hist = nsga_res.history
    surrogate_hist = surrogate_res.history

    nsga_n_evals,surrogate_n_evals = [],[]
    nsga_hist_F,surrogate_hist_F = [],[]
    for algo in nsga_hist:
        nsga_n_evals.append(algo.evaluator.n_eval)
        opt = algo.opt
        feas = np.where(opt.get("feasible"))[0]
        nsga_hist_F.append(opt.get("F")[feas])
    for algo in surrogate_hist:
        surrogate_n_evals.append(algo.evaluator.n_eval)
        opt = algo.opt
        feas = np.where(opt.get("feasible"))[0]
        surrogate_hist_F.append(opt.get("F")[feas])
    
    nsga_y, surrogate_y = [], []
    
    if criterion=='igd':
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
        # calculate the igd
        metric = IGD(parato_front, zero_to_one=True)
        nsga_y = [metric.do(_F) for _F in nsga_hist_F]
        surrogate_y = [metric.do(_F) for _F in surrogate_hist_F]

    elif criterion=='hv':
        nsga_F = nsga_res.opt.get("F")
        nsga_approx_ideal, nsga_approx_nadir = nsga_F.min(axis=0), nsga_F.max(axis=0)

        surrogate_F = surrogate_res.opt.get("F")
        surrogate_approx_ideal, surrogate_approx_nadir = surrogate_F.min(axis=0), surrogate_F.max(axis=0)

        approx_ideal, approx_nadir = np.minimum(nsga_approx_ideal, surrogate_approx_ideal), np.maximum(nsga_approx_nadir, surrogate_approx_nadir)
        metric = Hypervolume(ref_point= np.array([1.1, 1.1]),
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=approx_ideal,
                     nadir=approx_nadir)
        
        nsga_y = [metric.do(_F) for _F in nsga_hist_F]
        surrogate_y = [metric.do(_F) for _F in surrogate_hist_F]

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
    plt.plot(nsga_n_evals, nsga_y, color='g', label='NSGA2')
    plt.plot(surrogate_n_evals, surrogate_y, color='r', label='Surrogate')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
