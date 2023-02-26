from sklearn.ensemble import RandomForestClassifier
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.survival import Survival

from utils import get_non_dominated_solutions


# define the algorithm
class SurrogateNSGA2(NSGA2):

    def __init__(self, classifier_name=None, **kwargs):
        super().__init__(**kwargs)
        self.classifier_name = classifier_name

    def _initialize(self):
        super()._initialize()
        self.survival = SurrogateSurvival(self.survival, self.classifier_name)

class SurrogateSurvival(Survival):

    def __init__(self, survival, classifier_name=None):
        super().__init__()
        self.survival = survival
        self.classifier_name = classifier_name
        self.P_good = []
        self.P_bad = []

    def _do(self, problem, pop, _parents=None, **kwargs):
        if self.classifier_name is None:
            return self.survival._do(problem, pop, _parents, **kwargs)
        elif self.classifier_name == "RandomForest":
            self.classifier = RandomForestClassifier()
        
        # update P_good and P_bad
        Q = np.array(pop)
        F = pop.get("F")
        n = len(Q)
        _, idx = get_non_dominated_solutions(F, return_index=True)
        Q_good = Q[idx]
        Q_bad = np.array([ind for ind in Q if ind not in Q_good])
        all_good = np.concatenate([self.P_good, Q_good], axis=0)
        _, idx = get_non_dominated_solutions([ind.F for ind in all_good], return_index=True)
        all_good = all_good[idx]
        all_bad = np.concatenate([self.P_bad, self.P_good, Q], axis=0)
        all_bad = np.array([ind for ind in all_bad if ind not in all_good])
        self.P_good = np.random.choice(all_good, size=min(5*n, len(all_good)), replace=False)
        self.P_bad = np.random.choice(all_bad, size=min(5*n, len(all_bad)), replace=False)

        # predict the labels of offspring
        X = pop.get("X")
        train_x = np.concatenate([[ind.X for ind in self.P_good], [ind.X for ind in self.P_bad]], axis=0)
        train_y = np.concatenate([np.ones(len(self.P_good)), -1*np.ones(len(self.P_bad))], axis=0)
        self.classifier.fit(train_x, train_y)
        y_pred = self.classifier.predict(X)

        # select only the good offspring
        good_offspring = pop[y_pred == 1]

        return self.survival._do(problem, good_offspring, _parents, **kwargs)