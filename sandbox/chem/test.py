import unittest

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 

from ann import ANN
from utils import resample


class Tests(unittest.TestCase):

    def test_resample(self):
        X, y = make_classification(n_classes=2, class_sep=2,
                weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X, y)
        self.assertEqual(Counter(y_resampled), {0: 900, 1: 900})





if __name__ == '__main__':
    unittest.main(verbosity=3)