from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from surrogate import *
from kriging import *


def get_algorithm_name(name):
    algo = {
        'baseline-nsga2': NSGA2,
        'baseline-nsga3': NSGA3,
        'baseline-smsemoa': SMSEMOA,
        'sur-nocrowd-nsga2': SurrogateNSGA2,
        'sur-nocrowd-nsga3': SurrogateNSGA3,
        'sur-nocrowd-smsemoa': SurrogateSMSEMOA,
        'sur-nsga2': SurrogateNSGA2,
        'sur-nsga3': SurrogateNSGA3,
        'sur-smsemoa': SurrogateSMSEMOA,
        'krvea': KrigingRVEA,
        'moeadega': KrigingMOEAD,
    }
    return algo[name]


def get_algorithm(name, **kwargs):
    algorithm_class = get_algorithm_name(name)
    return algorithm_class(**kwargs)


def run_algorithm(problem, name):
    # define parameters
    pop_size = 50
    n_offsprings = 100
    n_eval = 550
    max_eval = 5
    termination = get_termination("n_eval", n_eval)
    ref_dirs = None
    if 'naga3' in name:
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
    elif 'rvea' in name or 'moead' in name:
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=15)
    do_crowding=True
    if 'nocrowd' in name:
        do_crowding=False

    # define the surrogate model
    classifier_name = {
        'dominance':"CART", 
        'crowding':"CART"
        }
    classifier_arg = {
        'dominance':{'max_depth': None, 'min_samples_split': 5}, 
        'crowding':{'max_depth': 20, 'min_samples_split': 4}
        }
    # classifier_name = "SVM"
    # classifier_arg={'kernel': 'rbf', 'C': 1, 'gamma': 0.1}
    # classifier_name = "KNN"
    # classifier_arg={'n_neighbors': 3}
    # classifier_name = "CART"
    # classifier_arg={'max_depth': None, 'min_samples_split':2}
    
    # define the algorithm
    if 'baseline' in name:
        algorithm = get_algorithm_name(name)(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            sampling=BinaryRandomSampling(),
            crossover=PointCrossover(n_points=2),
            mutation=BitflipMutation(),
            ref_dirs=ref_dirs,
            eliminate_duplicates=True
        )
    else:
        if 'moead' in name:
            algorithm = get_algorithm_name(name)(
            sampling=BinaryRandomSampling(),
            crossover=PointCrossover(n_points=2),
            mutation=BitflipMutation(),
            ref_dirs=ref_dirs,
            classifier_name=classifier_name,
            classifier_arg=classifier_arg,
            max_eval=max_eval,
            do_crowding=do_crowding
        )
        else:
            algorithm = get_algorithm_name(name)(
                pop_size=pop_size,
                n_offsprings=n_offsprings,
                sampling=BinaryRandomSampling(),
                crossover=PointCrossover(n_points=2),
                mutation=BitflipMutation(),
                eliminate_duplicates=True,
                ref_dirs=ref_dirs,
                classifier_name=classifier_name,
                classifier_arg=classifier_arg,
                max_eval=max_eval,
                do_crowding=do_crowding
            )
    
    # run the optimization
    res = minimize(problem,
            algorithm,
            termination = termination,
            save_history=True)
    
    return res
