import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tools import add_constant
from sklearn.linear_model import LogisticRegression

'''
implement probit regression, returns array of probits given exogenous variables

docs:
https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.Probit.html
https://www.statsmodels.org/dev/generated/statsmodels.tools.tools.add_constant.html

first step in 2-step Heckman correction process
'''
def probit_regression(X: np.array, y: np.array) -> np.array:
    #add constant for probit regression
    X = add_constant(X)

    #fit probit model
    mod = Probit(y, X)
    probit_mod = mod.fit()
    print(probit_mod.summary())

    y_probit = probit_mod.apply(X)
    return y_probit

#tests assumption in Heckman correction that errors are normal
def test_error_normality(x: np.array):
    pass

def heckman_bias_correction(X: np.array, y: np.array):
    pass
