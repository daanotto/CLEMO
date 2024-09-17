### IMPORT PACKAGES ###
import numpy as np
import scipy.spatial as sp
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid


### MAIN FUNCTIONS XOR ###
'''
Sample pertubations of instance: normally around original value
Args:     orig              - array of original instance parameters
          ftr_index_list    - list with indices of originial array to perturb
          model_lcl         - black-box model to explain
          method            - method to perturb instance. Only option now: normal
          var               - variance for pertubation
          size              - number of perturbed instances
          feasibility_check - check whether perturbed instance is feasible
          bounded_check     - check whether perturbed instance is bounded
Output:   org_plus_smpl  - array with rows original instance and size perturbed instances
'''

def sample_perturbations_normal(orig, ftr_index_list, model_lcl, hyperprm = {}, mean = 0, var = 0.2, size = 1000, feasibility_check = True, bounded_check = True):
    
    org_plus_prtb = [orig]
    cntr = 1
    incr = 1

    while cntr < size:
        orig_with_noise = copy.deepcopy(orig)
        good_sample = True
        
        for j in range(len(orig)):
            if j in ftr_index_list:
                lcl_var = orig_with_noise[j] * var
                orig_with_noise[j] = orig_with_noise[j] + np.random.normal(mean, lcl_var)

        if feasibility_check:
            good_sample = good_sample * model_lcl(orig_with_noise, output = 'feasibility', **hyperprm)
        if bounded_check:
            good_sample = good_sample * model_lcl(orig_with_noise, output = 'bounded', **hyperprm)

        if good_sample == True:
            org_plus_prtb.append(np.asarray(orig_with_noise))
            cntr  = cntr + 1
        
        incr = incr + 1

        if incr > size and cntr < size/2:
            raise ValueError("Too many unbounded or unfeasible samples, change sampling method")
        
    org_plus_prtb = np.asarray(org_plus_prtb)

    return org_plus_prtb

# Perturb instance parameters between -/+ epsilon
def sample_perturbations_epsilon(orig, ftr_index_list, model_lcl, hyperprm = {}, epsilon = 1, size = 1000, feasibility_check = True, bounded_check = True):
    
    org_plus_prtb = [orig]
    prtb = np.zeros(len(size),len(ftr_index_list))
    cntr = 1
    incr = 1

    while cntr < size:
        orig_with_noise = copy.deepcopy(orig)
        
        for j in range(len(orig)):
            if j in ftr_index_list:
                lmb = np.random.uniform(-1, 1)
                orig_with_noise[j] = orig_with_noise[j] + lmb * epsilon
                prtb[cntr][j] = lmb

        if feasibility_check:
            good_sample = good_sample * model_lcl(orig_with_noise, output = 'feasibility', **hyperprm)
        if bounded_check:
            good_sample = good_sample * model_lcl(orig_with_noise, output = 'bounded', **hyperprm)

        if good_sample == True:
            org_plus_prtb.append(np.asarray(orig_with_noise))
            cntr  = cntr + 1
        
        incr = incr + 1

        if incr > size and cntr < size/2:
            raise ValueError("Too many unbounded or unfeasible samples, change sampling method")
        
    org_plus_prtb = np.asarray(org_plus_prtb)
    prtb = np.asarray(prtb)

    return org_plus_prtb, prtb

def sample_perturbations_binary(orig, ftr_index_list, model_lcl, hyperprm = {}, prob = 0.8 , size = 1000, feasibility_check = True, bounded_check = True):
    
    org_plus_prtb = [orig]
    cntr = 1
    incr = 1

    while cntr < size:
        orig_with_noise = copy.deepcopy(orig)
        good_sample = True
        
        for j in range(len(orig)):
            if j in ftr_index_list:
                orig_with_noise[j] = np.random.binomial(n = 1, p=prob)

        if feasibility_check:
            good_sample = good_sample * model_lcl(orig_with_noise, output = 'feasibility', **hyperprm)
        if bounded_check:
            good_sample = good_sample * model_lcl(orig_with_noise, output = 'bounded', **hyperprm)

        if good_sample == True:
            org_plus_prtb.append(np.asarray(orig_with_noise))
            cntr  = cntr + 1
        
        incr = incr + 1

        if incr > size and cntr < size/2:
            raise ValueError("Too many unbounded or unfeasible samples, change sampling method")
        
    org_plus_prtb = np.asarray(org_plus_prtb)

    return org_plus_prtb



# Perturb instance parameters by scaling with factor between lower_bound and upper_bound
def sample_perturbations_scalar(orig, ftr_index_list, model_lcl, hyperprm = {}, lower_bound = 0.2, upper_bound = 2, size = 1000, feasibility_check = True, bounded_check = True):

    org_plus_prtb = [orig]
    prtb = np.ones(len(size),len(ftr_index_list))
    cntr = 1
    incr = 1

    while cntr < size:
        orig_with_noise = copy.deepcopy(orig)
        
        for j in range(len(orig)):
            if j in ftr_index_list:
                lmb = np.random.uniform(lower_bound, upper_bound)
                orig_with_noise[j] = orig_with_noise[j] * lmb 
                prtb[cntr][j] = lmb

        if feasibility_check:
            good_sample = good_sample * model_lcl(orig_with_noise, output = 'feasibility', **hyperprm)
        if bounded_check:
            good_sample = good_sample * model_lcl(orig_with_noise, output = 'bounded', **hyperprm)

        if good_sample == True:
            org_plus_prtb.append(np.asarray(orig_with_noise))
            cntr  = cntr + 1
        
        incr = incr + 1

        if incr > size and cntr < size/2:
            raise ValueError("Too many unbounded or unfeasible samples, change sampling method")
        
    org_plus_prtb = np.asarray(org_plus_prtb)
    prtb = np.asarray(prtb)
    
    return org_plus_prtb, prtb


'''
Determine weights of samples using distance with respect to original instance (the one to explain)
Args:   smpls               - array of perturbed samples of instance, including the original instance as first element
        ftr_index_list      - list with indices of originial array used as feature
        function            - function to determine weights. If None we defer to RBF-kernel and Euclidean distance
        width               - width of RBF kernel. If none, then 0.75 * #features used
Output: weights             - array with weights corresponding to samples
'''
    
def std_weight_function(a, b, ftr_index_list, kernel_width = None):
    d = np.linalg.norm(a - b)
    if kernel_width is None:
        krnl_wdth = 0.75 * len(ftr_index_list)
    else:
        krnl_wdth = kernel_width
    return np.exp(-(d ** 2) / (2* krnl_wdth ** 2))
    
def get_weights_from_samples(smpls, ftr_index_list, function = None, width = None):
    
    org = smpls[0]
    weights = []

    for smpl in smpls:
        if function is not None:
            weights.append(function(org, smpl))
        else:
            weights.append(std_weight_function(org, smpl, ftr_index_list, width))

    return weights

'''
Determine values of samples using black-box model 
Args:   smpls               - array of perturbed samples of instance
        model_lcl           - black-box optimization model
Output: values              - array with values to explain corresponding to samples
'''

def get_values_from_samples(smpls, model_lcl, hyperprm = {}):
    values = []
    
    for smpl in smpls:
        values.append(model_lcl(smpl, **hyperprm))
    
    return values

'''
Evaluate different kind of white-box models on different set of hyperparameters
Args:   model_type          - typpe of white-box model (restricted to specified set)
        hyper_prm_dct       - dictionary of hyperparameter values of model
        ?_train             - trains sample features/black-box outcome values/weights
        ?_test              - test sample features/black-box outcome values/weights
        store_all           - binary value. if True, store results of all combinations of hyperparameters. if False, store only hyperparemeters with best fit on train set
Output: rtrn_dict           - dictionary containing model fit information
'''

def model_search(model_type, hyper_prm_dct, X_train, X_test, Y_train, Y_test, W_train, W_test, store_all = False):
    
    best_perf = np.inf
    best_srgt = np.nan
    best_prms = np.nan
    rtrn_dict = {}

    hyper_prm_grid = ParameterGrid(hyper_prm_dct)
    for hyper_prm_set in hyper_prm_grid:

        if model_type == 'DecisionTreeRegressor':
            surrogate = DecisionTreeRegressor(random_state=42, **hyper_prm_set)
        elif model_type == 'DecisionTreeClassifier':
            surrogate = DecisionTreeClassifier(random_state=42, **hyper_prm_set)
        elif model_type == 'LinearRegression':
            surrogate = LinearRegression(**hyper_prm_set)
        elif model_type == 'RidgeRegression':
            surrogate = Ridge(random_state=42, **hyper_prm_set)
        elif model_type == 'LASSORegression':
            surrogate = Lasso(random_state=42, **hyper_prm_set)
        elif model_type == 'SVM':
            surrogate = svm(random_state=42, **hyper_prm_set)
        elif model_type == 'LogisticRegression':
            surrogate = LogisticRegression(random_state=42, **hyper_prm_set)
        else:
            raise ValueError("Model type not yet supported, please choose from: DecisionTreeRegressor, DecisionClassifier, LinearRegression, RidgeRegression, LASSORegression, SVM, LogisticRegression")

        surrogate.fit(X_train, Y_train, sample_weight= W_train)
        Y_pred_train = surrogate.predict(X_train)
        train_err = mean_squared_error(Y_train, Y_pred_train, sample_weight= W_train)

        if store_all == True:
            hyper_prm_str = str(hyper_prm_set)
            if hyper_prm_str == '':
                hyper_prm_str = ' '
            rtrn_dict[hyper_prm_str] = {}
            rtrn_dict[hyper_prm_str]['Model'] = surrogate
            Y_pred_test = surrogate.predict(X_test)
            if model_type in ['DecisionTreeRegressor', 'DecisionTreeClassifier']:
                rtrn_dict[hyper_prm_str]['Best model feature importance'] = surrogate.feature_importances_
            elif model_type == 'LogisticRegression':
                rtrn_dict[hyper_prm_str]['Best model coefficients'] = surrogate.coef_[0]
                rtrn_dict[hyper_prm_str]['Best model feature importance'] = np.abs(surrogate.coef_[0] * X_train.std(axis=0))/max(np.sum(np.abs(surrogate.coef_[0] * X_train.std(axis=0))), 0.00000001)
            else:
                rtrn_dict[hyper_prm_str]['Best model coefficients'] = surrogate.coef_
                rtrn_dict[hyper_prm_str]['Best model feature importance'] = np.abs(surrogate.coef_ * X_train.std(axis=0))/max(np.sum(np.abs(surrogate.coef_ * X_train.std(axis=0))), 0.00000001)
                rtrn_dict[hyper_prm_str]['R2'] = r2_score(Y_test, Y_pred_test)
            rtrn_dict[hyper_prm_str]['Wmse'] = mean_squared_error(Y_test, Y_pred_test, sample_weight= W_test)
            rtrn_dict[hyper_prm_str]['mse']  = mean_squared_error(Y_test, Y_pred_test)
            rtrn_dict[hyper_prm_str]['WL1e'] = mean_absolute_error(Y_test, Y_pred_test, sample_weight= W_test)
            rtrn_dict[hyper_prm_str]['L1e']  = mean_absolute_error(Y_test, Y_pred_test)
            rtrn_dict[hyper_prm_str]['Y_pred_test']  = Y_pred_test
            rtrn_dict[hyper_prm_str]['Y_test']  = Y_test

        if train_err < best_perf and store_all == False:
            best_perf = train_err
            best_srgt = surrogate
            best_prms = hyper_prm_set
    
    if store_all == False:
        Y_pred_test = best_srgt.predict(X_test)
        rtrn_dict['Model'] = best_srgt
        rtrn_dict['Best hyperparameters'] = best_prms
        if model_type in ['DecisionTreeRegressor', 'DecisionTreeClassifier']:
            rtrn_dict['Best model feature importance'] = best_srgt.feature_importances_
        elif model_type == 'LogisticRegression':
            rtrn_dict['Best model coefficients'] = best_srgt.coef_[0]
            rtrn_dict['Best model feature importance'] = np.abs(best_srgt.coef_[0] * X_train.std(axis=0))/max(np.sum(np.abs(best_srgt.coef_[0] * X_train.std(axis=0))), 0.00000001)
        else:
            rtrn_dict['Best model coefficients'] = best_srgt.coef_
            rtrn_dict['Best model feature importance'] = np.abs(best_srgt.coef_ * X_train.std(axis=0))/max(np.sum(np.abs(best_srgt.coef_ * X_train.std(axis=0))), 0.00000001)
            rtrn_dict['R2'] = r2_score(Y_test, Y_pred_test)
        rtrn_dict['Wmse'] = mean_squared_error(Y_test, Y_pred_test, sample_weight= W_test)
        rtrn_dict['mse']  = mean_squared_error(Y_test, Y_pred_test)
        rtrn_dict['WL1e'] = mean_absolute_error(Y_test, Y_pred_test, sample_weight= W_test)
        rtrn_dict['L1e']  = mean_absolute_error(Y_test, Y_pred_test)
        rtrn_dict['Y_pred_test']  = Y_pred_test
        rtrn_dict['Y_test']  = Y_test

    return rtrn_dict

'''
Split samples in train and test set to evaluate candidate white-box models
Args:   pos_mdls            - dictionary with white-box model types as keys and hyperparameter dictionary as values
        X, Y, W             - samples, black-box model values, weights
        feature_indices     - list of indices corresponding to features used
        train_part          - percentage of training data 
        store_all           - binary value. if True, store results of all combinations of hyperparameters. if False, store only hyperparemeters with best fit on train set
Output: outcome_dict        - dictionary containing model fit information per white-box-model type
'''

def train_test_explanation_models(pos_mdls, X, Y, W, feature_indices, train_part = 0.8, store_all = False):

    if len(feature_indices) > 1:
        X_train, X_test, Y_train, Y_test, W_train, W_test = train_test_split(X[:,feature_indices], Y, W,
                                                                                train_size = train_part, 
                                                                                test_size = 1-train_part, 
                                                                                random_state = 100)
    else:
        X_train, X_test, Y_train, Y_test, W_train, W_test = train_test_split(X, Y, W,
                                                                                train_size = train_part, 
                                                                                test_size = 1-train_part, 
                                                                                random_state = 100)
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
    
    outcome_dict = {}
    min_w = 1.0e-8
    if np.sum(W_train) <= min_w or np.sum(W_test) <= min_w:
        print("Neighborhood too small, too many weights zero")
        return {}

    for model_type in pos_mdls.keys():

        hyper_prm_dict = pos_mdls[model_type]
        outcome_dict[model_type] = model_search(model_type, hyper_prm_dict, X_train, X_test, Y_train, Y_test, W_train, W_test, store_all)
            
    return outcome_dict

'''
Plot feature importance/tree
Args:   solution_dict       - dictionary with white-box model types as keys and hyperparameter dictionary as values
        features_lbls       - labels of features
        surrogate_type      - list of model types to include in plot=
Output: outcome_dict        - dictionary containing model fit information per white-box-model type
'''

def plot_explanation_tree(solution_dict, features_lbls, surrogate_type = None):
    if surrogate_type not in ['DecisionTreeRegressor', 'DecisionClassifier']:
        raise ValueError("surrogate_type should be: 'DecisionTreeRegressor' or 'DecisionClassifier'")
    else:
        plt.figure()
        plot_tree(solution_dict[surrogate_type]['Model'], filled=True, feature_names = features_lbls)
        plt.show()
    return


def plot_explanation_feature_importance(solution_dict, features_lbls, surrogate_types = None):
    ind = np.arange(len(features_lbls))
    width = 1/len(surrogate_types)

    fig = plt.figure()
    fig, ax = plt.subplots()
    for i in range(len(surrogate_types)):
        ftr_prm = solution_dict[surrogate_types[i]]['Best model feature importance']
        ftr_imp = np.abs(ftr_prm)/np.sum(np.abs(ftr_prm))
        ax.barh(ind +i* width, ftr_imp, width, label=surrogate_types[i])

    ax.set(yticks=ind + 0.5, yticklabels=features_lbls, ylim=[2*width - 1, len(features_lbls)])
    if len(surrogate_types) > 1:
        ax.legend()
        plt.title('Relative feature importance for different surrogate types.')
    else:
        plt.title('Relative feature importance for ' + surrogate_types[0] + '  surrogate')
    plt.show()
    return






