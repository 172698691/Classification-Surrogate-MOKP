from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from pymoo.core.survival import Survival
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import NSGA2, calc_crowding_distance
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA

from utils import get_non_dominated_solutions


class SurrogateSurvival(Survival):

    def __init__(self, survival, classifier_name=dict(), classifier_arg=dict(), max_eval=5, do_crowding=True, has_opt=False):
        super().__init__()
        self.survival = survival
        self.classifier = {}
        self.classifier_name = classifier_name
        self.classifier_arg = classifier_arg
        self.max_eval = max_eval
        self.has_opt = has_opt
        self.do_crowding = do_crowding
        self.archive = []

    def _do(self, problem, pop, _parents=None, n_survive=None, **kwargs):
        if self.classifier_name['dominance'] is None:
            return self.survival._do(problem, pop, _parents, **kwargs)
        for key, value in self.classifier_name.items():
            self.classifier[key] = self.get_classifier(value, self.classifier_arg[key])
        
        # define population and offspring
        pop = kwargs['algorithm'].pop
        off = kwargs['algorithm'].off
        
        # update archive
        for ind in pop:
            if ind not in self.archive:
                self.archive.append(ind)
        
        # classify the offspring according to dominance relationship
        good_off = self.classify_dominance(off)
        
        # classify the offspring according to crowding distance
        if self.do_crowding and len(good_off) > self.max_eval:
            # good_off = self.classify_crowding(pop, good_off)
            good_off = self.classify_crowding(np.array(self.archive), good_off)
        
        # sort good_offspring accroding to pred_prob in descending order
        # good_X = good_offspring.get("X")
        # pred_prob = self.classifier.predict_proba(good_X)
        # good_offspring = good_offspring[np.argsort(pred_prob[:, 1])[::-1]]
        # good_offspring = good_offspring[:self.max_eval]

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
    
    def get_classifier(self, classifier_name, classifier_arg):
        if classifier_name == "RF":
            return RandomForestClassifier(**classifier_arg)
        elif classifier_name == "GB":
            return GradientBoostingClassifier(**classifier_arg)
        elif classifier_name == "SVM":
            return SVC(**classifier_arg)
        elif classifier_name == "KNN":
            return KNeighborsClassifier(**classifier_arg)
        elif classifier_name == "CART":
            return DecisionTreeClassifier(**classifier_arg)
        else:
            raise ValueError("Classifier name not recognized")
        
    def classify_dominance(self, off):
        # divide archive into good and bad
        F = [ind.F for ind in self.archive]
        _, idx = get_non_dominated_solutions(F, return_index=True)
        A_good = np.array(self.archive)[idx]
        A_bad = np.array([ind for ind in self.archive if ind not in A_good])
        if len(A_good) == 0 or len(A_bad) == 0:
            return off

        # train classifier
        train_x = np.concatenate([[ind.X for ind in A_good], [ind.X for ind in A_bad]], axis=0)
        train_y = np.concatenate([np.ones(len(A_good)), -1*np.ones(len(A_bad))], axis=0)
        self.classifier['dominance'].fit(train_x, train_y)

        # predict the labels of offspring
        off_X = off.get("X")
        y_pred = self.classifier['dominance'].predict(off_X)

        # select only the good offspring
        good_off = off[y_pred == 1]
        if len(good_off) == 0:
            good_off = off
        
        return good_off
    
    def classify_crowding(self, pop, off):
        # calculate crowding distance of pop
        # good pop
        # F = [ind.F for ind in pop]
        # F, idx = get_non_dominated_solutions(F, return_index=True)
        # pop = np.array(pop)[idx]
        # all pop
        F = np.array([ind.F for ind in pop])
        crowding_distance = calc_crowding_distance(F, filter_out_duplicates=False)

        # filter out the the ind with crowding distance = inf
        filter_crowding_distance = crowding_distance[crowding_distance != np.inf]
        filter_pop = pop[crowding_distance != np.inf]

        # divide P_good and P_bad according to mean crowding distance
        mean_crowding_distance = np.mean(filter_crowding_distance)
        P_good = filter_pop[filter_crowding_distance > mean_crowding_distance]
        P_bad = filter_pop[filter_crowding_distance <= mean_crowding_distance]
        if len(P_good) == 0 or len(P_bad) == 0:
            return off

        # train classifier
        train_x = np.concatenate([[ind.X for ind in P_good], [ind.X for ind in P_bad]], axis=0)
        train_y = np.concatenate([np.ones(len(P_good)), -1*np.ones(len(P_bad))], axis=0)
        self.classifier['crowding'].fit(train_x, train_y)

        # predict the labels offspring
        off_X = off.get("X")
        y_pred = self.classifier['crowding'].predict(off_X)

        # select only the good offspring
        good_off = off[y_pred == 1]
        if len(good_off) == 0:
            good_off = off
        
        return good_off


class SurrogateNSGA2(NSGA2):

    def __init__(self, classifier_name=dict(), classifier_arg=dict(), max_eval=5, do_crowding=True, **kwargs):
        super().__init__(**kwargs)
        self.classifier_name = classifier_name
        self.classifier_arg = classifier_arg
        self.max_eval = max_eval
        self.do_crowding = do_crowding

    def _initialize(self):
        super()._initialize()
        self.survival = SurrogateSurvival(self.survival, self.classifier_name, self.classifier_arg, self.max_eval, self.do_crowding)

    def advance(self, infills=None, **kwargs):
        if self.evaluator.n_eval > self.pop_size:
            self.evaluator.n_eval = self.evaluator.n_eval - self.n_offsprings + self.max_eval
        return super().advance(infills, **kwargs)


class SurrogateNSGA3(NSGA3):

    def __init__(self, classifier_name=dict(), classifier_arg=dict(), max_eval=5, do_crowding=True, **kwargs):
        super().__init__(**kwargs)
        self.classifier_name = classifier_name
        self.classifier_arg = classifier_arg
        self.max_eval = max_eval
        self.do_crowding = do_crowding

    def _initialize(self):
        super()._initialize()
        self.survival = SurrogateSurvival(self.survival, self.classifier_name, self.classifier_arg, self.max_eval, self.do_crowding, has_opt=True)
    
    def advance(self, infills=None, **kwargs):
        if self.evaluator.n_eval > self.pop_size:
            self.evaluator.n_eval = self.evaluator.n_eval - self.n_offsprings + self.max_eval
        return super().advance(infills, **kwargs)


class SurrogateRVEA(RVEA):

    def __init__(self, classifier_name=dict(), classifier_arg=dict(), max_eval=5, **kwargs):
        super().__init__(**kwargs)
        self.classifier_name = classifier_name
        self.classifier_arg = classifier_arg
        self.max_eval = max_eval

    def _initialize(self):
        super()._initialize()
        self.survival = SurrogateSurvival(self.survival, self.classifier_name, self.classifier_arg, self.max_eval)


class SurrogateSMSEMOA(SMSEMOA):

    def __init__(self, classifier_name=dict(), classifier_arg=dict(), max_eval=5, do_crowding=True, **kwargs):
        super().__init__(**kwargs)
        self.classifier_name = classifier_name
        self.classifier_arg = classifier_arg
        self.max_eval = max_eval
        self.do_crowding = do_crowding

    def _initialize(self):
        super()._initialize()
        self.survival = SurrogateSurvival(self.survival, self.classifier_name, self.classifier_arg, self.max_eval, self.do_crowding)

    def advance(self, infills=None, **kwargs):
        if self.evaluator.n_eval > self.pop_size:
            self.evaluator.n_eval = self.evaluator.n_eval - self.n_offsprings + self.max_eval
        return super().advance(infills, **kwargs)
    