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
from utils import *

class Resampler:
    def __init__(self, target_dataset: pd.DataFrame = None, reference_dataset: pd.DataFrame = None) -> None:
        pass

    def train_dt(self, dataset: pd.DataFrame, target_col: str, reference_cols: list, measure_impurity=False) -> None:
        #train initial decision tree
        random_state = 100
        tree = DecisionTreeClassifier.fit(dataset[reference_cols], dataset[target_col], random_state=random_state, min_impurity_decrease=0.01)
        self.dt = tree

        #obtain probability mass functions of leaf nodes
        self.leaf_node_pmfs = node_distribution(self.dt, dataset, target_col, reference_cols)

        if measure_impurity:
            #evaluate impurity reduction by decision tree
            start_impurity, end_impurity = self.evaluate_dt(dataset, target_col, reference_cols)
            self.start_impurity = start_impurity
            self.end_impurity = end_impurity
            print('starting Gini impurity:', start_impurity)
            print('ending Gini impurity:', end_impurity)

    def evaluate_dt(self, dataset: pd.DataFrame, target_col: str, reference_cols: list) -> tuple:
        start_impurity = impurity_from_pd_series(dataset[target_col])
        end_impurity = impurity_full_dt(self.tree, dataset, reference_cols)

        return start_impurity, end_impurity
    
    def resample_from_dataset(self, reference_dataset: pd.DataFrame, orig_reference_cols: list) -> pd.DataFrame:
        reference_dataset['node_id'] = self.dt.apply(reference_dataset[orig_reference_cols])
        reference_dataset['pred_target'] = reference_dataset['node_id'].apply(lambda x: mapped_pmf_sampling(x, self.leaf_node_pmfs))

        return reference_dataset

    def resample_from_descriptive_stats(self):
        pass

