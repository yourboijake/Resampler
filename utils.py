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

#accepts probability mass function as key value dict, where key = discrete value, value = probability assigned to value
def sample_from_pmf(prob_dict: dict) -> str:
    choices = np.array(list(prob_dict.keys()))
    probs = np.array(list(prob_dict.values()))
    return np.random.choice(choices, p=probs)

def mapped_pmf_sampling(node_id: str, pmf_dict: dict):
    return sample_from_pmf(pmf_dict[node_id])

def node_distribution(dt: DecisionTreeClassifier, dataset: pd.DataFrame, target_col: str, reference_cols: list) -> dict:
    pred_node_ids = dt.apply(dataset[reference_cols])
    dataset['node_id'] = pred_node_ids

    #target columns should be set to string, to ensure proper mapping in sample_from_pmf
    dataset['target'] = dataset['target'].astype(str)

    #convert into dictionary:
    #keys are node ids, values are dictionaries of {target_val: pmf probability}
    target_grouped_counts = dataset.groupby(['node_id', target_col]).count().iloc[:, 0]
    target_grouped_counts.name = 'node_target_count'
    target_grouped_counts = target_grouped_counts.reset_index()

    node_grouped_counts = dataset.groupby(['node_id']).count().iloc[:, 0]
    node_grouped_counts.name = 'node_count'
    node_grouped_counts = node_grouped_counts.reset_index()

    grouped_counts = target_grouped_counts.merge(node_grouped_counts, on='node_id', how='outer')
    grouped_counts['perc'] = grouped_counts['node_target_count'] / grouped_counts['node_count']
    grouped_counts.set_index(['node_id', target_col], inplace=True)

    # complicated: flattens grouped counts into nested dictionary
    pmf_dict = {node: grouped_counts.perc.xs(node).to_dict() for node in grouped_counts.index.levels[0]}

    return pmf_dict