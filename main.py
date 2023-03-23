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
    # set running times
    n_runs = 8

    # set plot data
    nsga_y_all, surrogate_y_all = [], []

    # loop
    for i in range(n_runs):

        # set config
        criterion = 'hv'

        # define the problem
        n_items = 300
        values = np.random.randint(1, 50, size=n_items)
        volume = np.random.randint(1, 30, size=n_items)
        weights = np.random.randint(1, 20, size=n_items)
        capacity = 0.6*np.sum(weights)
        problem = Knapsack(values, volume, weights, capacity)
        pop_size = 50
        max_eval = 20

        # define the surrogate model
        # classifier_name = "GB"
        # classifier_arg={'n_estimators': 100, 'learning_rate': 0.15, 'max_depth': 5}
        classifier_name = "RF"
        classifier_arg={'n_estimators': 250, 'max_depth': 15, 'min_samples_split':3}
        # classifier_name = "SVM"
        # classifier_arg={'kernel': 'rbf', 'C': 1, 'gamma': 0.1}
        # classifier_name = "KNN"
        # classifier_arg={'n_neighbors': 3}
        # classifier_name = "CART"
        # classifier_arg={'max_depth': None, 'min_samples_split':2}

        # define the algorithm
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True
        )
        surrogate_algorithm = SurrogateNSGA2(
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True,
            arc=True,
            classifier_name=classifier_name,
            classifier_arg=classifier_arg,
            max_eval=max_eval
        )

        # define the termination criterion
        termination = get_termination("n_eval", 500)

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

        # add init point
        # nsga_n_evals.insert(0, 0)
        # surrogate_n_evals.insert(0, 0)
        # init_x = np.random.random((pop_size, n_items))
        # init_x = (init_x < 0.5).astype(bool)
        # init_F = problem.evaluate_x(init_x)
        # nsga_hist_F.insert(0, init_F)
        # surrogate_hist_F.insert(0, init_F)
        
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
        
        # add to plot data
        nsga_y_all.append(nsga_y)
        surrogate_y_all.append(surrogate_y)

        # done loop
        print(f'Done loop {i+1}!')

    # plot res.F
    # plt.scatter(surrogate_res.F[:, 0], surrogate_res.F[:, 1], color='r')
    # plt.scatter(res.F[:, 0], res.F[:, 1], color='g')
    # plt.show()

    # print mean +- std
    print(f'NSGA2: {np.mean(nsga_y_all, axis=0)[-1]:.4f} +- {np.std(nsga_y_all, axis=0)[-1]:.4f}')
    print(f'Surrogate: {np.mean(surrogate_y_all, axis=0)[-1]:.4f} +- {np.std(surrogate_y_all, axis=0)[-1]:.4f}')

    # get mean
    nsga_y_all = np.mean(nsga_y_all, axis=0)
    surrogate_y_all = np.mean(surrogate_y_all, axis=0)

    # plot igd
    plt.plot(nsga_n_evals, nsga_y_all, color='b', label='NSGA2')
    plt.plot(surrogate_n_evals, surrogate_y_all, color='orange', label='Surrogate')
    plt.axhline(1, color="r", linestyle="--")
    plt.legend()
    plt.title(f'{classifier_arg}')
    plt.ylim(0.4, 1.05)
    plt.savefig('result.png')
    plt.show()


if __name__ == "__main__":
    main()
