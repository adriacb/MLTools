import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from pandas.api.types import CategoricalDtype

from sklearn.model_selection import KFold, StratifiedKFold

# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

# Mute warnings
warnings.filterwarnings('ignore')

def load_data():
    # Read data
    data_dir = Path("../data/")
    df_train = pd.read_csv(data_dir / "train.csv", index_col="Id")
    df_test = pd.read_csv(data_dir / "test.csv", index_col="Id")
    print(f"Train dataset has {df_train.shape[0]} rows and {df_train.shape[1]} columns.")
    print(f"Test dataset has {df_test.shape[0]} rows and {df_test.shape[1]} columns.")
    
    #all the data
    df = pd.concat([df_train, df_test])
    return df_train, df_test, df


def plot_cross_val(n_splits: int, splitter_func, df: pd.DataFrame, target_column: str, title_text: str) -> None:
    """Function to plot the cross-validation of various sklearn splitter objects."""
    split = 1
    plot_data = []
    
    if splitter_func == StratifiedKFold:
        y = df[target_column]
        X = df.drop(columns=[target_column])
        for train_index, valid_index in splitter_func(n_splits=n_splits).split(X, y):
            plot_data.append([train_index, 'Train', f'{split}'])
            plot_data.append([valid_index, 'Test', f'{split}'])
            split += 1
    else:
        for train_index, valid_index in splitter_func(n_splits=n_splits).split(df):
            plot_data.append([train_index, 'Train', f'{split}'])
            plot_data.append([valid_index, 'Test', f'{split}'])
            split += 1

    plot_df = pd.DataFrame(plot_data, columns=['Index', 'Dataset', 'Split']).explode('Index')

    plt.figure(figsize=(10, 6))
    handles = []
    labels = []
    
    train_handles = plt.scatter([], [], label='Train', color='blue', alpha=0.7)
    test_handles = plt.scatter([], [], label='Test', color='goldenrod', alpha=0.7)
    
    for split_num, group in plot_df.groupby('Split'):
        train_indices = group['Index'][group['Dataset'] == 'Train']
        test_indices = group['Index'][group['Dataset'] == 'Test']

        plt.scatter(train_indices, [split_num] * len(train_indices), color='blue', alpha=0.7)
        plt.scatter(test_indices, [split_num] * len(test_indices), color='goldenrod', alpha=0.7)
    
    handles.extend([train_handles, test_handles])
    labels.extend(['Train', 'Test'])

    plt.xlabel('Index')
    plt.ylabel('Split')
    plt.title(title_text)
    plt.legend(handles, labels)
    plt.grid(True)
    plt.show()

def describe_dataset(data:pd.DataFrame) -> None:
  print(data.info())
  print(data.describe())
  print()
  print("Null values in the dataframe:\n")
  print(data.isnull().sum())
  print("NaN values in the dataframe:\n")
  print(data.isna().sum())

def histogram(data, title='', xlabel='', ylabel='', bins=10, color='grey', edgecolor='black', alpha=0.7):
    """
    Generate a histogram using matplotlib.

    Parameters:
    - data: List or array-like data to create the histogram.
    - title: Title of the histogram (default is an empty string).
    - xlabel: Label for the x-axis (default is an empty string).
    - ylabel: Label for the y-axis (default is an empty string).
    - bins: Number of bins in the histogram (default is 10).
    - color: Color of the bars (default is 'skyblue').
    - edgecolor: Color of the edges of the bars (default is 'black').
    - alpha: Transparency of the bars (default is 0.7).

    Returns:
    - None (displays the histogram).
    """
    plt.hist(data, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def catplot(data, col):
    plt.figure(figsize=(10, 6))

    # Create a countplot to display the bars
    ax = sns.countplot(x=col, data=data, color="grey", order=data[col].value_counts().index)

    # Add count labels inside the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, color='black')

    plt.title(f'Count of {col}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()

def plot_boxplot_with_mean(data, xlabel='X Axis', ylabel='Y Axis', title='Boxplot with Mean'):
    plt.figure(figsize=(8, 4))
    ax = plt.gca()

    # Boxplot
    box_props = dict(color='black', facecolor='grey', linewidth=1.5)
    box = ax.boxplot(x=data.values, vert=False, widths=0.6, patch_artist=True, boxprops=box_props)
    for median in box['medians']:
        median.set(color='black', linewidth=2)

    # Mean line
    stats = data.describe()
    plt.axvline(stats["mean"], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.2f}', zorder=3)

    # Labels and title
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    # Customize y-axis ticks and labels
    plt.yticks([])  # Hide y-axis ticks

    # Add legend
    plt.legend()

    # Improve layout
    plt.tight_layout()

    # Show plot
    plt.show()

def plot_boxplot_with_mean_with_marginal_histogram(data, xlabel='X Axis', ylabel='Y Axis', title='Boxplot with Mean and Marginal Histogram', bins=10, color='grey', edgecolor='black', alpha=0.7):
    """
    Generate a boxplot with mean line and an adjacent marginal histogram using seaborn and matplotlib.

    Parameters:
    - data: List or array-like data to create the boxplot, mean line, and marginal histogram.
    - xlabel: Label for the x-axis (default is 'X Axis').
    - ylabel: Label for the y-axis (default is 'Y Axis').
    - title: Title of the plot (default is 'Boxplot with Mean and Marginal Histogram').
    - bins: Number of bins in the histogram (default is 10).
    - color: Color of the bars (default is 'grey').
    - edgecolor: Color of the edges of the bars (default is 'black').
    - alpha: Transparency of the bars (default is 0.7).

    Returns:
    - None (displays the boxplot with mean line and marginal histogram).
    """
    # Step 1: Create the figure and subplots
    fig, (hist_ax, box_ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 4]}, figsize=(10, 6))

    # Step 2: Plot the marginal histogram on the first subplot
    sns.histplot(data, bins=bins, kde=True, color=color, edgecolor=edgecolor, alpha=alpha, ax=hist_ax)
    hist_ax.set_ylabel('Density')
    hist_ax.set_title('Marginal Histogram')

    # Step 3: Remove the x-axis label from the histogram subplot
    hist_ax.set_xlabel('')  # Remove x-axis label

    # Step 4: Plot the boxplot with mean line on the second subplot
    sns.boxplot(data=data, ax=box_ax, orient='h', color=color, boxprops=dict(facecolor='grey'))
    mean_val = data.mean()
    box_ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}', zorder=3)
    box_ax.set_xlabel(xlabel)
    box_ax.set_ylabel(ylabel)
    box_ax.set_title(title)

    # Step 5: Customize the plot labels, titles, and other formatting
    hist_ax.yaxis.set_ticklabels([])  # Hide y-axis labels for the histogram
    hist_ax.yaxis.set_ticks([])       # Hide y-axis ticks for the histogram
    box_ax.legend()

    # Step 6: Show the plot
    plt.tight_layout()
    plt.show()

def diverging_bar_plot(data, labels, xlabel='$X$', ylabel='$Y$', title='', figsize=(14, 10), dpi=80, grid=True, grid_alpha=0.5, **kwargs):
    """
    Function to generate a diverging bar plot.
    
    Parameters:
        data: pandas DataFrame or Series, containing the data to plot.
        labels: list or array-like, labels for each bar.
        xlabel: str, label for the x-axis.
        ylabel: str, label for the y-axis.
        title: str, title of the plot.
        figsize: tuple, size of the figure (width, height) in inches.
        dpi: int, dots per inch.
        grid: bool, whether to show grid lines or not.
        grid_alpha: float, transparency of grid lines (0 to 1).
        **kwargs: additional keyword arguments passed to plt.barh().

    Returns:
        None (displays the plot).
    """
    plt.figure(figsize=figsize, dpi=dpi)
    plt.hlines(y=range(len(data)), xmin=0, xmax=data, **kwargs)
    plt.gca().set(ylabel=ylabel, xlabel=xlabel)
    plt.yticks(range(len(data)), labels, fontsize=12)
    plt.title(title, fontdict={'size': 20})
    
    if grid:
        plt.grid(linestyle='--', alpha=grid_alpha)
    
    plt.show()

def kdeplot_by_class(data, x, hue, xlabel='', ylabel='Density', title='', figsize=(10, 6), linewidth=2.5, **kwargs):
    """
    Function to generate KDE plot by different classes.

    Parameters:
        data: pandas DataFrame, containing the data.
        x: str, variable for the x-axis.
        hue: str, variable for coloring different classes.
        xlabel: str, label for the x-axis.
        ylabel: str, label for the y-axis.
        title: str, title of the plot.
        figsize: tuple, size of the figure (width, height) in inches.
        linewidth: float, width of the KDE curve.
        **kwargs: additional keyword arguments passed to sns.kdeplot().

    Returns:
        None (displays the plot).
    """
    plt.figure(figsize=figsize)
    sns.kdeplot(data=data, x=x, hue=hue, linewidth=linewidth, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title=hue)
    plt.show()


def plot_categorical_grid(
        df: pd.DataFrame, 
        cols_per_row:int=3, 
        figsize:tuple=(15, 10), 
        max_categories_for_labels:int=10
        ):
    """
    Plot a grid of categorical columns from a DataFrame.

    Parameters:
    - df: DataFrame containing categorical columns to be plotted.
    - cols_per_row: Number of columns to be displayed per row.
    - figsize: Tuple specifying the size of the figure (width, height).
    - max_categories_for_labels: Maximum number of categories for which labels will be added.

    Returns:
    - None (displays the grid plot)
    """

    # Select only the categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # Check if there are any categorical columns left to plot
    if not cat_cols:
        print("No categorical columns to plot.")
        return

    # Calculate the number of rows needed based on the number of columns per row
    num_rows = len(cat_cols) // cols_per_row
    num_rows += 1 if len(cat_cols) % cols_per_row != 0 else 0

    # Create subplots
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=figsize)
    fig.subplots_adjust(hspace=0.5)

    # Flatten axes if there is only one row
    if num_rows == 1:
        axes = [axes]

    # Plot each categorical column
    for i, col_name in enumerate(cat_cols):
        row = i // cols_per_row
        col = i % cols_per_row
        
        # Calculate percentages
        cat_counts = df[col_name].value_counts()
        total_count = cat_counts.sum()
        cat_percentages = cat_counts / total_count * 100
        
        # Create bar plot
        sns.barplot(x=cat_percentages.index, y=cat_percentages.values, ax=axes[row][col], color='grey')
        axes[row][col].set_title(col_name)
        axes[row][col].set_ylabel('Percentage')
        
        # Rotate x-axis labels if the number of categories is below the threshold
        if len(cat_percentages) <= max_categories_for_labels:
            axes[row][col].tick_params(axis='x', rotation=45)
        else:
            axes[row][col].set_xticklabels([])
        
        # Add percentage labels if the number of categories is less than the threshold
        if len(cat_percentages) <= max_categories_for_labels:
            for index, value in enumerate(cat_percentages):
                axes[row][col].text(index, value, f'{value:.2f}%', ha='center', va='bottom')

    # Remove empty subplots if any
    if len(cat_cols) < num_rows * cols_per_row:
        for i in range(len(cat_cols), num_rows * cols_per_row):
            fig.delaxes(axes.flatten()[i])

    plt.show()

def plot_categorical_vs_continuous(df:pd.DataFrame, label_col:str = 'label', cols_per_row:int=4):
    # Identify categorical and continuous columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    continuous_columns = df.select_dtypes(include=['int64', 'float64']).columns

    # Iterate over each categorical column
    for cat_col in categorical_columns:
        num_plots = len(continuous_columns)
        rows = (num_plots // cols_per_row) + (num_plots % cols_per_row > 0)

        # Create a figure for the current categorical column
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 5, rows * 5))
        axes = axes.flatten()

        # Plot each continuous column against the current categorical column
        for plot_idx, cont_col in enumerate(continuous_columns):
            sns.boxplot(x=df[cat_col], y=df[cont_col], ax=axes[plot_idx], hue=df[label_col], color='grey')
            #sns.stripplot(x=df[cat_col], y=df[cont_col], ax=axes[plot_idx], jitter=True, color='grey')
            #sns.catplot(x=df[cat_col], y=df[cont_col], kind='boxen', ax=axes[plot_idx], color='grey')
            #sns.catplot(x=df[cat_col], y=df[cont_col], kind='swarm', hue=df[label_col], ax=axes[plot_idx], color='grey')
            axes[plot_idx].set_title(f'{cat_col} vs {cont_col}')
            axes[plot_idx].set_xticklabels(axes[plot_idx].get_xticklabels(), rotation=45)

        # Remove any empty subplots
        for i in range(plot_idx + 1, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.suptitle(f'Plots for {cat_col}', y=1.02)
        plt.show()

def plot_grid_of_boxplots(df, color='grey'):
    """
    Generate a grid of boxplots with mean lines for all continuous variables in the DataFrame.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - xlabel: Label for the x-axis (default is 'X Axis').
    - ylabel: Label for the y-axis (default is 'Y Axis').
    - title: Title of the plot (default is 'Boxplot with Mean').
    - color: Color of the boxplot (default is 'grey').

    Returns:
    - None (displays the grid of plots).
    """
    # Step 1: Identify continuous variables
    continuous_vars = df.select_dtypes(include=['int64', 'float64']).columns

    # Step 2: Determine the grid size
    num_vars = len(continuous_vars)
    grid_size = int(np.ceil(np.sqrt(num_vars)))
    
    # Step 3: Create the figure with subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(continuous_vars):
        # Step 4: Create subplots for boxplot
        box_ax = axes[i]
        
        data = df[var].dropna()
        
        # Plot the boxplot with mean line
        sns.boxplot(data=data, ax=box_ax, orient='h', color=color, boxprops=dict(facecolor='grey'))
        mean_val = data.mean()
        box_ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}', zorder=3)
        box_ax.set_xlabel(f"{var}")
        box_ax.set_title(f'{var}')
        box_ax.legend()
    
    # Remove any unused axes
    for j in range(num_vars, len(axes)):
        fig.delaxes(axes[j])
    
    # Step 5: Adjust layout
    plt.tight_layout()
    plt.show()