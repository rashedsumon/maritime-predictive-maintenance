# src/viz_utils.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_distribution(df: pd.DataFrame, feature: str):
    """Return a matplotlib figure for Streamlit to render."""
    fig, ax = plt.subplots(figsize=(6, 3))
    df[feature].dropna().hist(bins=30, ax=ax)
    ax.set_title(f'Distribution: {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('count')
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, labels=None):
    import numpy as np
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion matrix")
    tick_marks = range(len(cm))
    if labels is not None:
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, format(cm[i][j], 'd'), ha="center", va="center", color="white" if cm[i][j] > cm.max()/2 else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    return fig
