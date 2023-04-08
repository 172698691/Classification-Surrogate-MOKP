from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from pymoo.core.survival import Survival
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA

from utils import get_non_dominated_solutions


class SurrogateSurvival(Survival):

    def __init__(self, survival, classifier_name=None, classifier_arg=dict(), max_eval=5, has_opt=False):
        super().__init__()
        self.survival = survival
        self.classifier_name = classifier_name
        self.classifier_arg = classifier_arg
        self.max_eval = max_eval
        self.has_opt = has_opt
        self.archive = []

    def _do(self, problem, pop, _parents=None, n_survive=None, **kwargs):
        if self.classifier_name is None:
            return self.survival._do(problem, pop, _parents, **kwargs)
        elif self.classifier_name == "RF":
            self.classifier = RandomForestClassifier(**self.classifier_arg)
        elif self.classifier_name == "GB":
            self.classifier = GradientBoostingClassifier(**self.classifier_arg)
        elif self.classifier_name == "SVM":
            self.classifier = SVC(**self.classifier_arg)
        elif self.classifier_name == "KNN":
            self.classifier = KNeighborsClassifier(**self.classifier_arg)
        elif self.classifier_name == "CART":
            self.classifier = DecisionTreeClassifier(**self.classifier_arg)
        else:
            raise ValueError("Classifier name not recognized")
        
        # update archive
        for ind in kwargs['algorithm'].pop:
            if ind not in self.archive:
                self.archive.append(ind)
        
        # divide archive into good and bad
        F = [ind.F for ind in self.archive]
        _, idx = get_non_dominated_solutions(F, return_index=True)
        A_good = np.array(self.archive)[idx]
        A_bad = np.array([ind for ind in self.archive if ind not in A_good])

        # train classifier
        train_x = np.concatenate([[ind.X for ind in A_good], [ind.X for ind in A_bad]], axis=0)
        train_y = np.concatenate([np.ones(len(A_good)), -1*np.ones(len(A_bad))], axis=0)
        self.classifier.fit(train_x, train_y)

        # predict the labels of offspring
        X = pop.get("X")
        y_pred = self.classifier.predict(X)

        # select only the good offspring
        good_offspring = pop[y_pred == 1]
        if len(good_offspring) == 0:
            good_offspring = pop
        good_offspring = good_offspring[np.random.choice(len(good_offspring), size=min(len(good_offspring), self.max_eval), replace=False)]

        # do the original survival
        if self.has_opt:
            self.survival._do(problem=problem, pop=good_offspring, _parents=_parents, n_survive=n_survive, **kwargs)
            self.opt = self.survival.opt
            return self.opt
        
        return self.survival._do(problem=problem, pop=good_offspring, _parents=_parents, n_survive=n_survive, **kwargs)

    def adapt(self):
        self.survival.adapt()


class SurrogateNSGA2(NSGA2):

    def __init__(self, classifier_name=None, classifier_arg=dict(), max_eval=5, **kwargs):
        super().__init__(**kwargs)
        self.classifier_name = classifier_name
        self.classifier_arg = classifier_arg
        self.max_eval = max_eval

    def _initialize(self):
        super()._initialize()
        self.survival = SurrogateSurvival(self.survival, self.classifier_name, self.classifier_arg, self.max_eval)

    def advance(self, infills=None, **kwargs):
        if self.evaluator.n_eval > self.pop_size:
            self.evaluator.n_eval = self.evaluator.n_eval - self.n_offsprings + self.max_eval
        return super().advance(infills, **kwargs)


class SurrogateNSGA3(NSGA3):

    def __init__(self, classifier_name=None, classifier_arg=dict(), max_eval=5, **kwargs):
        super().__init__(**kwargs)
        self.classifier_name = classifier_name
        self.classifier_arg = classifier_arg
        self.max_eval = max_eval

    def _initialize(self):
        super()._initialize()
        self.survival = SurrogateSurvival(self.survival, self.classifier_name, self.classifier_arg, self.max_eval, has_opt=True)


class SurrogateRVEA(RVEA):

    def __init__(self, classifier_name=None, classifier_arg=dict(), max_eval=5, **kwargs):
        super().__init__(**kwargs)
        self.classifier_name = classifier_name
        self.classifier_arg = classifier_arg
        self.max_eval = max_eval

    def _initialize(self):
        super()._initialize()
        self.survival = SurrogateSurvival(self.survival, self.classifier_name, self.classifier_arg, self.max_eval)