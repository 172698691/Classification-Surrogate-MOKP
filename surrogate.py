from sklearn.ensemble import RandomForestClassifier
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.survival import Survival


# define the algorithm
class SurrogateNSGA2(NSGA2):

    def __init__(self, classifier_name=None, **kwargs):
        super().__init__(**kwargs)
        self.classifier_name = classifier_name

    def _initialize(self):
        super()._initialize()
        self.survival = MySurvival(self.survival, self.classifier_name)

class MySurvival(Survival):

    def __init__(self, survival, classifier_name=None):
        super().__init__()
        self.survival = survival
        self.classifier_name = classifier_name

    def _do(self, problem, pop, _parents=None, **kwargs):
        if self.classifier_name is None:
            return self.survival._do(problem, pop, _parents, **kwargs)
        elif self.classifier_name == "RandomForest":
            self.classifier = RandomForestClassifier()

        # predict the labels of offspring
        X = pop.get("X")
        train_x = np.random.randint(low=0, high=2, size=(100, X.shape[-1]), dtype=bool)
        train_y = np.random.choice([-1, 1], size=(100,))
        self.classifier.fit(train_x, train_y)
        y_pred = self.classifier.predict(X)

        # select only the good offspring
        good_offspring = pop[y_pred == 1]

        return self.survival._do(problem, good_offspring, _parents, **kwargs)