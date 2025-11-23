# src/viz.py
from typing import List, Optional
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")  # global theme for consistent style

def plot_target(series: pd.Series, log: bool = False, savepath: Optional[str] = None, show: bool = True) -> None:
    """Plot distribution of target. Use `log=True` to visualize log-transformed distribution."""
    if series is None or len(series) == 0:
        raise ValueError("Empty series provided to plot_target.")
    s = series.dropna()
    if log:
        s = s.map(lambda x: None if x is None else (np.log(x) if x > 0 else np.nan)).dropna()

    plt.figure(figsize=(10, 4))
    sns.histplot(s, kde=True, bins=50, edgecolor="black")
    title_suffix = " (log)" if log else ""
    plt.title(f"Target distribution{title_suffix}")
    plt.xlabel(series.name or "value")
    plt.ylabel("count")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_correlation(df: pd.DataFrame, features: List[str], annot: bool = True, show: bool = True) -> None:
    """
    Plot pairwise Pearson correlation heatmap for given features.
    """
    correlation_matrix = df[features].corr(method='pearson')

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=annot,           # Write the data value in each cell
        fmt=".2f",            # String formatting code to use when adding annotations
        cmap='coolwarm',      # Diverging colormap (Red for pos, Blue for neg)
        vmin=-1, vmax=1,      # Anchor the colormap range
        center=0,             # Center the colormap at 0
        linewidths=.5,
        cbar_kws={'label': 'Correlation Coefficient'}
    )

    plt.title('Numerical Features Correlation Heatmap (Pearson)',
              fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()

def plot_value_counts(series: pd.Series, top_n: int = 20, show: bool = True) -> None:
    counts = series.value_counts().nlargest(top_n)
    plt.figure(figsize=(6, max(3, len(counts)*0.25)))
    sns.barplot(x=counts.values, y=counts.index)
    plt.title(f"Top {top_n} value counts: {series.name}")
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()

def plot_features(X: pd.DataFrame, numerical_features: list, categorical_features: list) -> None:
    """
    Generates histograms for numerical features and bar plots for categorical features
    to visualize data distribution.
    """
    n_num = len(numerical_features)
    n_cat = len(categorical_features)

    # Determine grid dimensions based on the number of features
    max_cols = max(n_num, n_cat, 1)

    # Dynamic figure height: Allocate 4 units for each row (numerical/categorical)
    fig_height = 4 * ((1 if n_num > 0 else 0) + (1 if n_cat > 0 else 0))

    fig = plt.figure(figsize=(max_cols * 5, fig_height))
    gs = fig.add_gridspec(2, max_cols, hspace=0.4, wspace=0.3)

    # Plot Numerical Features
    if n_num > 0:
        print(f"Plotting {n_num} numerical histograms...")
        for i, col in enumerate(numerical_features):
            ax = fig.add_subplot(gs[0, i])

            # Limit bins to 50 or the number of unique values to prevent over-plotting
            bins = min(X[col].nunique(), 50)
            # Enable Kernel Density Estimate (KDE) only if there are enough unique values
            kde = True if X[col].nunique() > 10 else False

            sns.histplot(X[col].dropna(), bins=bins, kde=kde, ax=ax, edgecolor='black')

            ax.set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')
            sns.despine(ax=ax) # Remove top and right spines for a cleaner look

    # Plot Categorical Features
    if n_cat > 0:
        print(f"Plotting {n_cat} categorical bar plots...")
        # If numerical features exist, plot categorical on the second row (index 1)
        row_idx = 1 if n_num > 0 else 0

        for i, col in enumerate(categorical_features):
            ax = fig.add_subplot(gs[row_idx, i])

            display_counts = X[col].value_counts()

            sns.barplot(x=display_counts.index, y=display_counts.values, ax=ax,
                        palette='viridis')

            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title(f'Count: {col}', fontsize=12, fontweight='bold')
            sns.despine(ax=ax)

    plt.suptitle('Feature Distribution Visualization', fontsize=16, fontweight='heavy', y=0.98)
    # Adjust subplots to fit into the figure area nicely
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()