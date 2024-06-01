import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

## Feature importance

# is a measure that assigns a score to each feature in the dataset
# based on how much it contributes to the model's prediction

def plot_tree_importance(model, feature_names, top_n=10):

    # make importances relative to max importance
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                        model.estimators_],
                                        axis = 0)
    
    plt.bar(range(top_n), feature_importance_normalized[:top_n], align = 'center')
    plt.xticks(range(top_n), feature_names[:top_n], rotation = 90)
    plt.tight_layout()
    plt.show()
    

def plot_correlation(df, annotate = True):
    """
    Is a statistical measure that expresses the strenght of the
    relation between two variables.
    
    corr = sum((x - x.mean()) * (y - y.mean())) / sqrt(sum((x - x.mean())**2) * sum((y - y.mean())**2))
    
    if the value is close to 1, the variables are positively correlated
    if the value is close to -1, the variables are negatively correlated
    
    """
    df = df.select_dtypes(include=['number'])
    # Check if there are non-numeric columns
    if len(df.columns) < len(df.columns):
        raise("Warning: Non-numeric columns were dropped for correlation calculation.")
    
    ax = sns.heatmap(df.corr(), annot = annotate)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title("Correlation plot")
    plt.show()
    
    
    
