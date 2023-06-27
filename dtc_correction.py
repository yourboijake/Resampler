'''
see notes for details

d1, target dataset = dataset 1 (biased sample, includes target value and population reference columns)
d2, reference dataset = dataset 2 (unbiased sample, includes just population reference columns)
desc_stats = dictionary, containing population reference data point, and stats characteristics, such as:
    - normal: mean and std dev
    - uniform: specify range
    - exponential: specify lambda
    - discrete_pmf: specify dictionary with values and distribution (must correspond to values in )
    
basic api structure:
- define class Resampler

Attributes:
- decision tree (defined after calling initial training function)
- any training hyperparameters?

Methods:
- train_dt(d1, target_col='', reference_cols=[]): trains initial decision tree using d1, optional hyperparameters
    - preprocess data (null values, data types, etc.)
    - figure out number of unique values in target class
    - set internal decision tree attribute
    - assess_dt, report results
    - return None
- assess_dt(d1):
    - internal function, estimates information gain (reduction in Shannon entropy) from tree
    - how well do the population reference features improve classification accuracy?
- resample_from_dataset(d2): runs d2 through decision tree, using leaf node randomness to estimate population target value
    - return distribution of population target value results
- resample_from_descriptive_stats(desc_stats, sample_size=1000):
    - for population reference feature in desc_stats, create sample_size samples
    - combine these together to create a synthetic d2
    - run resample_from_dataset(synthetic d2)
    - return distribution of population target value results
- monte_carlo_from_dataset(d2, iterations=100): randomly reruns resample_from_dataset(d2) iterations times
    - return ? (maybe averaged results plus a confidence interval?)
- monte_carlo_from_descriptive_stats(desc_stats, iterations=100): randomy reruns 
    - return ? (maybe averaged results plus a confidence interval?)

'''
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from utils import *
from functools import reduce
from collections import Counter

class Resampler:
    def __init__(self, target_dataset: pd.DataFrame = None, reference_dataset: pd.DataFrame = None) -> None:
        pass

    def train_dt(self, dataset: pd.DataFrame, target_col: str, reference_cols: list) -> None:
        #train initial decision tree using grid search CV
        param_grid = {"max_depth": [5, 10],
                      "min_impurity_decrease": [0.0, 0.2]}
        tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
        tree.fit(dataset[reference_cols], dataset[target_col])
        self.dt = tree.best_estimator_
        self.best_params = tree.best_params_

        #obtain probability mass functions of leaf nodes
        self.leaf_node_pmfs = node_distribution(self.dt, dataset, target_col, reference_cols)

        #evaluate impurity reduction by decision tree
        start_impurity, end_impurity = self.evaluate_dt(dataset, target_col, reference_cols)
        self.start_impurity = start_impurity
        self.end_impurity = end_impurity
        print('starting Gini impurity:', start_impurity)
        print('ending Gini impurity:', end_impurity)
        print("mean accuracy of resulting DTC on full dataset:", self.dt.score(dataset[reference_cols], dataset[target_col]))
    
    def get_best_grid_search_params(self):
        return self.best_params

    def evaluate_dt(self, dataset: pd.DataFrame, target_col: str, reference_cols: list) -> tuple:
        start_impurity = impurity_from_pd_series(dataset[target_col])
        end_impurity = impurity_full_dt(self.dt, dataset, reference_cols)

        return start_impurity, end_impurity
    
    def resample_from_dataset(self, reference_dataset: pd.DataFrame, orig_reference_cols: list) -> pd.DataFrame:
        reference_dataset['node_id'] = self.dt.apply(reference_dataset[orig_reference_cols])
        reference_dataset['pred_target'] = reference_dataset['node_id'].apply(lambda x: mapped_pmf_sampling(x, self.leaf_node_pmfs))

        return reference_dataset

    def resample_from_descriptive_stats(self, dist_dict: dict, orig_reference_cols: list, sample_size: int = 10000) -> pd.DataFrame:
        synthetic_dataset = {}
        for dist in dist_dict.keys():
            synthetic_dataset[dist] = sample_from_descriptive_stats(dist, dist_dict[dist], sample_size)
        
        synthetic_df = pd.DataFrame(synthetic_dataset)
        synthetic_df['node_id'] = self.dt.apply(synthetic_df[orig_reference_cols])
        synthetic_df['pred_target'] = synthetic_df['node_id'].apply(lambda x: mapped_pmf_sampling(x, self.leaf_node_pmfs))        
        
        return synthetic_df

    #returns tuple of pd.Series: mean of % of total for each class, and std dev of % of total for each class
    def monte_carlo_from_dataset(self, reference_dataset: pd.DataFrame, orig_reference_cols: list, iterations: int = 100) -> tuple:
        prob_dict = {}
        for i in range(1, iterations):
            iter_dict = dict(self.resample_from_dataset(reference_dataset, orig_reference_cols)['pred_target'].value_counts(normalize=True))
            for k, v in enumerate(iter_dict):
                if k not in prob_dict.keys():
                    prob_dict[k] = [v]
                else:
                    prob_dict[k].append(v)

        for k, v in enumerate(prob_dict):
            prob_dict[k] = np.array(v)

        prob_dict_mean = dict(zip(prob_dict.keys(), [np.mean(v) for v in prob_dict.values()]))
        prob_dict_std = dict(zip(prob_dict.keys(), [np.std(v) for v in prob_dict.values()]))

        return prob_dict_mean, prob_dict_std