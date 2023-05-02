import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression

from pymoo.core.survival import Survival
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.moead import MOEAD

from utils import get_non_dominated_solutions


class KrigingSurvival(Survival):

    def __init__(self, survival, max_eval=5, has_opt=False):
        super().__init__()
        self.survival = survival
        self.max_eval = max_eval
        self.has_opt = has_opt
        self.archive = []

    def _do(self, problem, pop, _parents=None, n_survive=None, **kwargs):
        # define population and offspring
        pop = kwargs['algorithm'].pop
        off = kwargs['algorithm'].off
        
        # update archive
        for ind in pop:
            if ind not in self.archive:
                self.archive.append(ind)
        
        # classify the offspring according to dominance relationship
        good_off = self.classify_dominance(pop, off)

        # select max_eval number of offspring randomly
        good_off = good_off[np.random.choice(len(good_off), size=min(len(good_off), self.max_eval), replace=False)]

        # merge population and good_offspring
        merged_pop = Population.merge(pop, good_off)

        # do the original survival
        if self.has_opt:
            self.survival._do(problem=problem, pop=merged_pop, _parents=_parents, n_survive=n_survive, **kwargs)
            self.opt = self.survival.opt
            return self.opt
        
        return self.survival._do(problem=problem, pop=merged_pop, _parents=_parents, n_survive=n_survive, **kwargs)

    def adapt(self):
        self.survival.adapt()
        
    def classify_dominance(self, pop, off):
        # Define the base models
        # kernel = RBF()
        # base_model = GaussianProcessRegressor(kernel=kernel)
        meta_model = LinearRegression(copy_X=True, fit_intercept=False)
        # meta_model = GaussianProcessRegressor()
        
        # define data
        X = np.array([ind.X for ind in pop])
        F = np.array([ind.F for ind in pop])

        # Train the base models
        # base_model.fit(X, F)
        # base_predictions = base_model.predict(X)

        # # Train the meta-model
        meta_model.fit(X, F)
        # meta_model.fit(X, F - base_predictions)

        # predict the labels of offspring
        off_X = off.get("X")
        # base_predictions_new = base_model.predict(off_X)
        # F_predictions = base_predictions_new + meta_model.predict(off_X)
        F_predictions = meta_model.predict(off_X)

        # select only the good offspring
        _, idx = get_non_dominated_solutions(F_predictions, return_index=True)
        good_off = off[idx]
        
        return good_off


class KrigingNSGA2(NSGA2):

    def __init__(self, max_eval=5, **kwargs):
        super().__init__(**kwargs)
        self.max_eval = max_eval

    def _initialize(self):
        super()._initialize()
        self.survival = KrigingSurvival(self.survival, self.max_eval)

    def advance(self, infills=None, **kwargs):
        if self.evaluator.n_eval > self.pop_size:
            self.evaluator.n_eval = self.evaluator.n_eval - self.n_offsprings + self.max_eval
        return super().advance(infills, **kwargs)


class KrigingRVEA(RVEA):

    def __init__(self, max_eval=5, **kwargs):
        super().__init__(**kwargs)
        self.max_eval = max_eval

    def _initialize(self):
        super()._initialize()
        self.survival = KrigingSurvival(self.survival, self.max_eval)
    
    # def advance(self, infills=None, **kwargs):
    #     if self.evaluator.n_eval > self.pop_size:
    #         self.evaluator.n_eval = self.evaluator.n_eval - self.n_offsprings + self.max_eval
    #     return super().advance(infills, **kwargs)


class KrigingMOEAD(MOEAD):

    def __init__(self, max_eval=5, **kwargs):
        super().__init__(**kwargs)
        self.max_eval = max_eval

    def _initialize(self):
        super()._initialize()
        self.survival = KrigingSurvival(self.survival, self.max_eval)
    
    # def advance(self, infills=None, **kwargs):
    #     if self.evaluator.n_eval > self.pop_size:
    #         self.evaluator.n_eval = self.evaluator.n_eval - self.n_offsprings + self.max_eval
    #     return super().advance(infills, **kwargs)
