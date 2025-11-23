# src/viz.py
from typing import List, Optional
import pandas as pd
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
    """Plot pairwise Pearson correlation heatmap for given features."""
    if not features:
        raise ValueError("features list is empty.")
    corr = df[features].corr()
    plt.figure(figsize=(max(8, len(features)*0.5), max(6, len(features)*0.4)))
    sns.heatmap(corr, annot=annot, fmt=".2f", center=0, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation matrix")
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
