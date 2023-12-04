import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, chi2, SelectPercentile, f_classif, SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier

## feature selection
# https://medium.com/@sumitb2015/complete-guide-on-feature-engineering-part-ii-8eeb06bee44d
def select_kbest(x: np.ndarray, y: np.ndarray, score_func: chi2, k:int = 5) -> np.ndarray:
    """
    Selects the top k features based on the specified score function.

    Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The target data.
        score_func (chi2): The score function to use for feature selection.
        k (int, optional): The number of features to select. Defaults to 5.

    Returns:
        np.ndarray: The selected features.
    """
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(x, y)
    x_selected = selector.transform(x)
    return x_selected
    
    
def select_percentile(x: np.ndarray, y: np.ndarray, score_func: f_classif, percentile: int = 50) -> np.ndarray:
    """
    Selects the top percentile features based on the specified score function.

    Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The target data.
        score_func (f_classif): The score function to use for feature selection.
        percentile (int, optional): The percentile of features to select. Defaults to 5.

    Returns:
        np.ndarray: The selected features.
    """
    selector = SelectPercentile(score_func=score_func, percentile=percentile)
    selector.fit(x, y)
    x_selected = selector.transform(x)
    return x_selected

def rfe(x: np.ndarray, y: np.ndarray, n_features_to_select: int = 5) -> np.ndarray:
    """
    Selects the top n_features_to_select features using recursive feature elimination.

    Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The target data.
        estimator (object): The estimator to use for feature selection.
        n_features_to_select (int, optional): The number of features to select. Defaults to 5.

    Returns:
        np.ndarray: The selected features.
    """
    
    estimator = LogisticRegression()
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
    selector.fit(x, y)
    x_selected = selector.transform(x)
    return x_selected

## Model-based feature selection

def select_lasso(x: np.ndarray, y: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Selects the features using Lasso regression.

    Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The target data.
        alpha (float, optional): The regularization parameter. Defaults to 0.5.

    Returns:
        np.ndarray: The selected features.
    """
    estimator = Lasso(alpha=alpha)
    selector = SelectFromModel(estimator=estimator)
    selector.fit(x, y)
    x_selected = selector.transform(x)
    return x_selected


## Tree-based feature selection

def select_rfc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Selects the features using Random Forest Classifier.

    Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The target data.

    Returns:
        np.ndarray: The selected features.
    """
    estimator = RandomForestClassifier()
    selector = SelectFromModel(estimator=estimator)
    selector.fit(x, y)
    x_selected = selector.transform(x)
    return x_selected

def seq_fs(x: np.ndarray, y: np.ndarray, n_features_to_select: int = 5, direction: str = 'forward') -> np.ndarray:
    """
    Selects the top n_features_to_select features using sequential feature selection.
    Evaluate subsets of features rather than individual features. They iteratively add or remove features to build a model.

    Parameters:
        x (np.ndarray): The input data.
        y (np.ndarray): The target data.
        n_features_to_select (int, optional): The number of features to select. Defaults to 5.
        direction (str, optional): The direction of feature selection. Defaults to 'forward'.
        
    Returns:
        np.ndarray: The selected features.
    """
    estimator = LogisticRegression()
    selector = SequentialFeatureSelector(estimator=estimator, n_features_to_select=n_features_to_select, direction=direction)
    selector.fit(x, y)
    x_selected = selector.transform(x)
    return x_selected


## dimensionality reduction

def pca():
    pass

def tsne():
    pass

def lda():
    pass

## Non-linear feature selection
def kpca()