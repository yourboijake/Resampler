import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def impurity_from_pd_series(ser):
    val_counts = ser.value_counts()
    impurity = val_counts.values / val_counts.values.sum()
    impurity = 1 - np.square(impurity).sum()
    return impurity

def impurity_from_dist(arr):
    pass

def impurity_full_dt(dt: DecisionTreeClassifier, dataset: pd.DataFrame, reference_cols: list) -> float:
    # get leaf node indices
    leaf_nodes_by_sample = dt.apply(dataset[reference_cols])

    # leaf node each sample belongs to
    leaf_nodes = np.unique(leaf_nodes_by_sample)

    # determine total gini impurity of decision tree (weighted average)
    tot_imp = 0.0
    num_sam = len(dataset.index)
    for node in leaf_nodes:
        nd_ct = dt.tree_.n_node_samples[node]  # num samples at 'node'
        tot_imp += (nd_ct/num_sam)*dt.tree_.impurity[node] # gini impurity at 'node'

    return tot_imp