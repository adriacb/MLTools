import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, chi2, SelectPercentile, f_classif, SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
## feature selection
# https://medium.com/@sumitb2015/complete-guide-on-feature-engineering-part-ii-8eeb06bee44d

def rfe_feature_selection(X_train, y_train, X_test, n_features=10, output_path="../prep_data/"):
    names = X_train.columns
    lr = LogisticRegression(max_iter=1000)
    rfe = RFE(lr, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    feature_subset = np.array(names)[rfe.get_support()]
    
    # Plotting the selected features
    selected_importances = np.where(rfe.get_support(), 1, 0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, selected_importances)
    plt.xticks(rotation=90)
    plt.grid()
    plt.title("Feature Selection Based on RFE")
    plt.xlabel("Features")
    plt.ylabel("Selected (1) / Not Selected (0)")
    plt.ylim(0, 1.1)
    plt.show()
    
    print("Selected Feature Columns: {}".format(feature_subset))
    
    selected_df_train = X_train[feature_subset]
    selected_df_test = X_test[feature_subset]
    
    selected_df_train.to_parquet(output_path + "X_train_rfe.parquet")
    selected_df_test.to_parquet(output_path + "X_test_rfe.parquet")
    
    return feature_subset

def select_kbest_feature_selection(X_train, y_train, X_test, n_features=10, output_path="../prep_data/"):
    names = X_train.columns
    skb = SelectKBest(score_func=f_classif, k=n_features)
    skb.fit(X_train, y_train)
    feature_subset = np.array(names)[skb.get_support()]
    
    # Plotting the selected features
    selected_importances = np.where(skb.get_support(), 1, 0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, selected_importances)
    plt.xticks(rotation=90)
    plt.grid()
    plt.title("Feature Selection Based on SelectKBest")
    plt.xlabel("Features")
    plt.ylabel("Selected (1) / Not Selected (0)")
    plt.ylim(0, 1.1)
    plt.show()
    
    print("Selected Feature Columns: {}".format(feature_subset))
    
    selected_df_train = X_train[feature_subset]
    selected_df_test = X_test[feature_subset]
    
    selected_df_train.to_parquet(output_path + "X_train_skb.parquet")
    selected_df_test.to_parquet(output_path + "X_test_skb.parquet")
    
    return feature_subset

    
    
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

def random_forest_feature_selection(X_train, y_train, X_test, rf_params={"n_estimators": [100, 200, 300], "max_features": ['auto', 'sqrt', 'log2']}, threshold=0.001, output_path="../prep_data/"):
    names = X_train.columns
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier()
    rf_cv = GridSearchCV(rf, param_grid=rf_params, cv=kf)
    rf_cv.fit(X_train, y_train)
    best_rf = rf_cv.best_estimator_
    importances = best_rf.feature_importances_
    feature_subset = np.array(names)[importances > threshold]
    
    # Plotting the feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(names, importances)
    plt.xticks(rotation=90)
    plt.grid()
    plt.title("Feature Selection Based on RandomForestClassifier")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.ylim(0, max(importances) * 1.1)
    plt.show()
    
    print("Selected Feature Columns: {}".format(feature_subset))
    
    selected_df_train = X_train[feature_subset]
    selected_df_test = X_test[feature_subset]
    
    selected_df_train.to_parquet(output_path + "X_train_rf.parquet")
    selected_df_test.to_parquet(output_path + "X_test_rf.parquet")
    
    return feature_subset

## Model-based feature selection

def lasso_feature_selection(X_train, y_train, X_test, alpha_range=np.arange(0.00001, 10, 500), threshold=0.001, output_path="../prep_data/"):
    names = X_train.columns
    params = {"alpha": alpha_range}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lasso = Lasso()
    lasso_cv = GridSearchCV(lasso, param_grid=params, cv=kf)
    lasso_cv.fit(X_train, y_train)
    best_alpha = lasso_cv.best_params_['alpha']
    model = Lasso(alpha=best_alpha)
    model.fit(X_train, y_train)
    importances = np.abs(model.coef_)
    feature_subset = np.array(names)[importances > threshold]
    
    # Plotting the feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(names, importances)
    plt.xticks(rotation=90)
    plt.grid()
    plt.title("Feature Selection Based on Lasso")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.ylim(0, max(importances) * 1.1)
    plt.show()
    
    print("Selected Feature Columns: {}".format(feature_subset))
    
    selected_df_train = X_train[feature_subset]
    selected_df_test = X_test[feature_subset]
    
    selected_df_train.to_parquet(output_path + "X_train_lasso.parquet")
    selected_df_test.to_parquet(output_path + "X_test_lasso.parquet")
    
    return feature_subset

def sfs_feature_selection(X_train, y_train, X_test, n_features=10, output_path="../prep_data/"):
    names = X_train.columns
    lr = LogisticRegression(max_iter=1000)
    sfs = SequentialFeatureSelector(lr, n_features_to_select=n_features)
    sfs.fit(X_train, y_train)
    feature_subset = np.array(names)[sfs.get_support()]
    
    # Plotting the selected features
    selected_importances = np.where(sfs.get_support(), 1, 0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, selected_importances)
    plt.xticks(rotation=90)
    plt.grid()
    plt.title("Feature Selection Based on SequentialFeatureSelector")
    plt.xlabel("Features")
    plt.ylabel("Selected (1) / Not Selected (0)")
    plt.ylim(0, 1.1)
    plt.show()
    
    print("Selected Feature Columns: {}".format(feature_subset))
    
    selected_df_train = X_train[feature_subset]
    selected_df_test = X_test[feature_subset]
    
    selected_df_train.to_parquet(output_path + "X_train_sfs.parquet")
    selected_df_test.to_parquet(output_path + "X_test_sfs.parquet")
    
    return feature_subset



## dimensionality reduction

def pca():
    pass

def tsne():
    pass

def lda():
    pass

## Non-linear feature selection
def kpca():
    pass