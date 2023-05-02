import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
# Set font family and font size for all text in the plot
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 3

from utils import *
from simpleProblem import *
from surrogate import *
from kriging import *
from run import *


def main():
    # set running times
    n_runs = 10

    # set algorithm
    name_list = ['baseline-nsga2', 'sur-nsga2']

    # set plot data
    y_all = [[]] * len(name_list)
    space_all = [[]] * len(name_list)

    # loop
    for n_run in range(n_runs):

        # set config
        criterion = 'hv'

        # define the problem
        n_knapsack = 1
        n_items = 300
        values_1 = np.random.randint(1, 50, size=n_items)
        values_2 = np.random.randint(1, 30, size=n_items)
        weights = np.random.randint(1, 20, size=n_items)
        capacities = np.random.randint(int(0.6/n_knapsack*np.sum(weights)), int(0.8/n_knapsack*np.sum(weights)), size=n_knapsack)
        # problem = Knapsack(values_1, values_2, weights, capacities)
        problem = Knapsack_one(values_1, values_2, weights, capacities)
        problem_unconstrain = Knapsack_unconstrained(values_1, values_2, weights, capacities)

        # run the algorithm
        res_list = []
        for name in name_list:
            while True:
                if 'moead' in name:
                    res = run_algorithm(problem=problem_unconstrain, name=name)
                else:
                    res = run_algorithm(problem=problem, name=name)
                if res.opt is not None:
                    res_list.append(res)
                    break
                print(f"{name} res.opt is None")

        # get the history
        hist_list = [res.history for res in res_list]
        hist_F_list = []
        n_evals_list = []
        for hist in hist_list:
            n_evals = []
            hist_F = []
            for algo in hist:
                n_evals.append(algo.evaluator.n_eval)
                opt = algo.opt
                feas = np.where(opt.get("feasible"))[0]
                hist_F.append(opt.get("F")[feas])
            n_evals_list.append(n_evals)
            hist_F_list.append(hist_F)
                
        if criterion == 'igd':
            pass
            # nsga_y, surrogate_y = cal_igd(problem, nsga_hist_F, surrogate_hist_F)
        elif criterion == 'hv':
            res_F_list = [res.opt.get("F") for res in res_list]
            y_list = cal_hv(res_F_list, hist_F_list)
        # add to plot data
        for i in range(len(y_list)):
            y_all[i].append(y_list[i])

        # calculate spacing
        # space_list = cal_spacing(res_list, hist_F_list)
        space_list = cal_pd(hist_F_list)
        for i in range(len(space_list)):
            space_all[i].append(space_list[i])

        # done loop
        print(f'Done loop {n_run+1}!')

    # print mean +- std
    for i in range(len(y_all)):
        print(f'{name_list[i]} HV: {np.mean(y_all[i], axis=0)[-1]:.4f} +- {np.std(y_all[i], axis=0)[-1]:.4f}')
    print()
    for i in range(len(space_all)):
        print(f'{name_list[i]} Spacing: {np.mean(space_all[i], axis=0)[-1]:.4f} +- {np.std(space_all[i], axis=0)[-1]:.4f}')

    # get mean
    y_all_mean, space_all_mean = [], []
    for i in range(len(y_all)):
        y_all_mean.append(np.mean(y_all[i], axis=0))
    for i in range(len(space_all)):
        space_all_mean.append(np.mean(space_all[i], axis=0))

    # plot igd
    plt.figure(figsize=(10, 8))
    plt.axhline(1, color="r", linestyle="--")
    for i in range(len(y_all_mean)):
        plt.plot(n_evals_list[i], y_all_mean[i], label=name_list[i])
        print(name_list[i], 'HV:')
        print(list(n_evals_list[i]))
        print(list(y_all_mean[i]))
    plt.legend()
    # plt.title(f'{classifier_arg}')
    plt.title('HV result')
    # plt.ylim(0.3, 1.05)
    plt.xlabel('Function evaluations', fontsize=25)
    plt.ylabel('HV', fontsize=25)
    # plt.savefig('result.png')
    plt.show()

    # plot spacing
    mean_mun = 5
    plt.figure(figsize=(10, 8))
    for i in range(len(space_all_mean)):
        if 'sur' not in name_list[i]:
            plt.plot(n_evals_list[i], space_all_mean[i], label=name_list[i])
            print(name_list[i], 'PD:')
            print(list(n_evals_list[i]))
            print(list(space_all_mean[i]))
        else:
            plt.plot(*get_mean_every_n(n_evals_list[i], space_all_mean[i], mean_mun), label=name_list[i])
            x, y = get_mean_every_n(n_evals_list[i], space_all_mean[i], mean_mun)
            print(name_list[i], 'PD:')
            print(list(x))
            print(list(y))
    plt.legend()
    # plt.title(f'{classifier_arg}')
    plt.title(f'PD result')
    # plt.ylim(0.3, 1.05)
    plt.xlabel('Function evaluations', fontsize=25)
    plt.ylabel('PD', fontsize=25)
    # plt.savefig('result.png')
    plt.show()


if __name__ == "__main__":
    main()
    # nohup python main.py > output.txt 2>&1 &
