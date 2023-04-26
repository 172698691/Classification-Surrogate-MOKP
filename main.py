import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
# Set font family and font size for all text in the plot
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 3

from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.ref_dirs import get_reference_directions

from utils import *
from simpleProblem import Knapsack
from surrogate import *


def cal_igd(problem, nsga_hist_F, surrogate_hist_F):
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


def main():
    # set running times
    n_runs = 8

    # set plot data
    nsga_y_all, surrogate_nocrowd_y_all, surrogate_y_all, archive_y_all = [], [], [], []
    nsga_space_all, surrogate_nocrowd_space_all, surrogate_space_all, archive_space_all = [], [], [], []

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
        n_offsprings = 50
        n_eval = 500
        max_eval = 5

        # define the surrogate model
        classifier_name = {
            'dominance':"RF", 
            'crowding':"RF"
            }
        classifier_arg = {
            'dominance':{'n_estimators': 300, 'max_depth': 12, 'min_samples_split': 3}, 
            'crowding':{'n_estimators': 200, 'max_depth': 8, 'min_samples_split': 3}
            }
        # classifier_name = "GB"
        # classifier_arg={'n_estimators': 100, 'learning_rate': 0.15, 'max_depth': 5}
        # classifier_name = "SVM"
        # classifier_arg={'kernel': 'rbf', 'C': 1, 'gamma': 0.1}
        # classifier_name = "KNN"
        # classifier_arg={'n_neighbors': 3}
        # classifier_name = "CART"
        # classifier_arg={'max_depth': None, 'min_samples_split':2}

        # define the algorithm
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
        algorithm = NSGA2(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            # ref_dirs=ref_dirs,
            eliminate_duplicates=True
        )
        surrogate_nocrowd_algorithm = SurrogateNSGA2(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True,
            # ref_dirs=ref_dirs,
            classifier_name=classifier_name,
            classifier_arg=classifier_arg,
            max_eval=max_eval,
            do_crowding=False
        )
        surrogate_algorithm = SurrogateNSGA2(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True,
            # ref_dirs=ref_dirs,
            classifier_name=classifier_name,
            classifier_arg=classifier_arg,
            max_eval=max_eval
        )

        # run the optimization
        nsga_res = minimize(problem,
                    algorithm,
                    termination = get_termination("n_eval", n_eval),
                    save_history=True)
        surrogate_nocrowd_res = minimize(problem,
                    surrogate_nocrowd_algorithm,
                    termination = get_termination("n_eval", n_eval),
                    save_history=True)
        surrogate_res = minimize(problem,
                    surrogate_algorithm,
                    termination = get_termination("n_eval", n_eval),
                    save_history=True)
        
        nsga_hist = nsga_res.history
        surrogate_nocrowd_hist = surrogate_nocrowd_res.history
        surrogate_hist = surrogate_res.history

        nsga_n_evals,surrogate_nocrowd_n_evals,surrogate_n_evals,archive_n_evals = [],[],[],[]
        nsga_hist_F,surrogate_nocrowd_hist_F,surrogate_hist_F,archive_hist_F = [],[],[],[]
        for algo in nsga_hist:
            nsga_n_evals.append(algo.evaluator.n_eval)
            opt = algo.opt
            feas = np.where(opt.get("feasible"))[0]
            nsga_hist_F.append(opt.get("F")[feas])
        for algo in surrogate_nocrowd_hist:
            surrogate_nocrowd_n_evals.append(algo.evaluator.n_eval)
            opt = algo.opt
            feas = np.where(opt.get("feasible"))[0]
            surrogate_nocrowd_hist_F.append(opt.get("F")[feas])
        for algo in surrogate_hist:
            surrogate_n_evals.append(algo.evaluator.n_eval)
            opt = algo.opt
            feas = np.where(opt.get("feasible"))[0]
            surrogate_hist_F.append(opt.get("F")[feas])
        for algo in surrogate_hist:
            archive_n_evals.append(algo.evaluator.n_eval)
            archive = algo.survival.archive
            F = [ind.F for ind in archive]
            _, idx = get_non_dominated_solutions(F, return_index=True)
            archive = np.array(archive)[idx]
            archive_hist_F.append(np.array([ind.F for ind in archive]))
        
        hist_F_list = [nsga_hist_F, surrogate_nocrowd_hist_F, surrogate_hist_F, archive_hist_F]
                
        if criterion == 'igd':
            nsga_y, surrogate_y = cal_igd(problem, nsga_hist_F, surrogate_hist_F)
        elif criterion == 'hv':
            res_list = [nsga_res, surrogate_nocrowd_res, surrogate_res]
            res_F_list = [res.opt.get("F") for res in res_list]
            archive = surrogate_res.algorithm.survival.archive
            F = [ind.F for ind in archive]
            _, idx = get_non_dominated_solutions(F, return_index=True)
            archive = np.array(archive)[idx]
            res_F_list.append(np.array([ind.F for ind in archive]))
            y_list = cal_hv(res_F_list, hist_F_list)
        # add to plot data
        nsga_y_all.append(y_list[0])
        surrogate_nocrowd_y_all.append(y_list[1])
        surrogate_y_all.append(y_list[2])
        archive_y_all.append(y_list[3])

        # calculate spacing
        # space_list = cal_spacing(res_list, hist_F_list)
        space_list = cal_pd(hist_F_list)
        nsga_space_all.append(space_list[0])
        surrogate_nocrowd_space_all.append(space_list[1])
        surrogate_space_all.append(space_list[2])
        archive_space_all.append(space_list[3])

        # done loop
        print(f'Done loop {i+1}!')

    # plot res.F
    # plt.scatter(surrogate_res.F[:, 0], surrogate_res.F[:, 1], color='r')
    # plt.scatter(res.F[:, 0], res.F[:, 1], color='g')
    # plt.show()

    # print mean +- std
    print(f'NSGA2 HV: {np.mean(nsga_y_all, axis=0)[-1]:.4f} +- {np.std(nsga_y_all, axis=0)[-1]:.4f}')
    print(f'Surrogate No_crowd HV: {np.mean(surrogate_nocrowd_y_all, axis=0)[-1]:.4f} +- {np.std(surrogate_nocrowd_y_all, axis=0)[-1]:.4f}')
    print(f'Surrogate Crowd HV: {np.mean(surrogate_y_all, axis=0)[-1]:.4f} +- {np.std(surrogate_y_all, axis=0)[-1]:.4f}')
    print(f'Archive HV: {np.mean(archive_y_all, axis=0)[-1]:.4f} +- {np.std(archive_y_all, axis=0)[-1]:.4f}')
    print()
    print(f'NSGA2 Spacing: {np.mean(nsga_space_all, axis=0)[-1]:.4f} +- {np.std(nsga_space_all, axis=0)[-1]:.4f}')
    print(f'Surrogate No_crowd Spacing: {np.mean(surrogate_nocrowd_space_all, axis=0)[-1]:.4f} +- {np.std(surrogate_nocrowd_space_all, axis=0)[-1]:.4f}')
    print(f'Surrogate Crowd Spacing: {np.mean(surrogate_space_all, axis=0)[-1]:.4f} +- {np.std(surrogate_space_all, axis=0)[-1]:.4f}')
    print(f'Archive Spacing: {np.mean(archive_space_all, axis=0)[-1]:.4f} +- {np.std(archive_space_all, axis=0)[-1]:.4f}')

    # get mean
    nsga_y_all = np.mean(nsga_y_all, axis=0)
    surrogate_nocrowd_y_all = np.mean(surrogate_nocrowd_y_all, axis=0)
    surrogate_y_all = np.mean(surrogate_y_all, axis=0)
    nsga_space_all = np.mean(nsga_space_all, axis=0)
    surrogate_nocrowd_space_all = np.mean(surrogate_nocrowd_space_all, axis=0)
    surrogate_space_all = np.mean(surrogate_space_all, axis=0)
    archive_y_all = np.mean(archive_y_all, axis=0)
    archive_space_all = np.mean(archive_space_all, axis=0)

    # plot igd
    plt.figure(figsize=(10, 8))
    plt.plot(nsga_n_evals, nsga_y_all, color='b', label='Baseline')
    plt.plot(surrogate_nocrowd_n_evals, surrogate_nocrowd_y_all, color='g', label='Surrogate No_crowd')
    plt.plot(surrogate_n_evals, surrogate_y_all, color='orange', label='Surrogate Crowd')
    plt.plot(archive_n_evals, archive_y_all, color='r', label='Archive')
    plt.axhline(1, color="r", linestyle="--")
    plt.legend()
    # plt.title(f'{classifier_arg}')
    plt.title(f'NSGA-2 Random {max_eval}')
    plt.ylim(0.3, 1.05)
    plt.xlabel('Function evaluations', fontsize=25)
    plt.ylabel('HV', fontsize=25)
    # plt.savefig('result.png')
    plt.show()

    # plot spacing
    plt.figure(figsize=(10, 8))
    plt.plot(nsga_n_evals, nsga_space_all, color='b', label='Baseline')
    plt.plot(surrogate_nocrowd_n_evals, surrogate_nocrowd_space_all, color='g', label='Surrogate No_crowd')
    plt.plot(surrogate_n_evals, surrogate_space_all, color='orange', label='Surrogate Crowd')
    plt.plot(archive_n_evals, archive_space_all, color='r', label='Archive')
    plt.legend()
    # plt.title(f'{classifier_arg}')
    plt.title(f'NSGA-2 Random {max_eval}')
    # plt.ylim(0.3, 1.05)
    plt.xlabel('Function evaluations', fontsize=25)
    plt.ylabel('PD', fontsize=25)
    # plt.savefig('result.png')
    plt.show()


if __name__ == "__main__":
    main()
