import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import seaborn as sns
import itertools
import os

from scipy import stats
from scipy.stats import chi2
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

import pickle

from config_logit import config

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)



class UnivariableAnalysis():
    """ Used to see the impact of each variable individually and remove 
    those that are irrelevant.

    """

    def __init__(self, config, X, y):
        # configuration
        self.pvalue_e = config["pvalue_e"]
        self.maxiter = config["maxiter"]
        self.variables = X.columns.values[1:]

        
        # intercept only
        model = sm.Logit(y, X['const'])
        result = model.fit(maxiter=self.maxiter)
        print(result.summary2())
        self.L0 = result.llnull 


        # Univariable Analysis
        self.vars_to_remove = list()
        pvalues = {}

        for var in self.variables:
            variables = ['const']+[var]
            model = sm.Logit(y, X[variables])
            result = model.fit(maxiter=self.maxiter)
            print(result.summary2())
            loglik = result.llf
            loglik_ratio = -2 * (self.L0 - loglik)
            # loglik_ratio = 2 * (loglik - self.L0)
            pvalue = chi2.sf(loglik_ratio, 1)
            if pvalue > self.pvalue_e:
                self.vars_to_remove.append(var)
            else:
                pvalues[var] = pvalue

        self.pvalues = pd.DataFrame.from_dict(pvalues, orient='index', columns = ['pvalues'])
        print(self.pvalues)

        print("VARIABLES TO BE REMOVED:")
        print(self.vars_to_remove)


class MultivariableAnalysis():
    """ Build a model either by forward or backward selection

    """

    def __init__(self, config):
        self.pvalue_e = config["pvalue_e"]
        self.pvalue_z = config["pvalue_z"]
        self.maxiter = config['maxiter']
        self.solver = config['solver']
        self.variables = list(X.columns.values)


    def stepwise_fwd_selection(self, X, y, starting_features):

        # FIRST VARIABLE 
        # _____________________________________________________________________
        # model with first variable chosen from univariable analysis
        base_var = starting_features
        result = self.logit_on_subset(y=y, X=X, feature_set=base_var)
        base_loglik = result['loglik']
        loglik_ratio = 2 * (base_loglik - result['L0'])

        # create a dictionary of pvalues for variables added to the model
        pvalues_in = {}
        for var in base_var:
            pvalues_in[var] = result['result'].pvalues.loc[var]

        # SUBSEQUENT VARIABLES
        # _____________________________________________________________________
        i = 1 # keep track of iteration
        go = True # stopping rule

        in_model = base_var

        while go:

            # forward step --------------------------------------------
            pvalues_fwd = {}
            for var in self.variables:
                if var not in in_model:
                    variables = in_model + [var]
                    result = self.logit_on_subset(y=y, X=X, feature_set=variables)
                    loglik_ratio = 2 * (result['loglik'] - base_loglik)
                    pvalues_fwd[var] = chi2.sf(loglik_ratio, 1)
            
            # variable with smallest pvalue 
            min_var = min(pvalues_fwd, key=lambda k: pvalues_fwd[k])
            pvalue_min = pvalues_fwd[min_var]

            # add to the model only if it respects the threshold else, model is complete
            if pvalue_min < self.pvalue_e:
                in_model.append(min_var)
                pvalues_in[min_var] = pvalue_min
            else: 
                go = False

            # new model before backward check
            result = self.logit_on_subset(y=y, X=X, feature_set=in_model)
            base_loglik = result['loglik']
            print("NEW MODEL (FORWARD):")
            print(result['result'].summary2())
            # ---------------------------------------------------------

            # backward check ------------------------------------------
            pvalues = result['result'].pvalues
            pvalues_out = pvalues.where(pvalues > self.pvalue_z)
            vars_to_remove = pvalues_out.index.tolist()
            in_model = [x for x in in_model if x is not vars_to_remove]

            # new model after backward check
            result = self.logit_on_subset(y=y, X=X, feature_set=in_model)
            base_loglik = result['loglik']
            print("NEW MODEL (BACKWARD):")
            print(result['result'].summary2())
            # ----------------------------------------------------------

            # end of loop
            print("ROUND: "+str(i))
            i = i+1


        print("FINAL VARIABLES:")
        print(in_model)
        print("FINAL VARIABLES P-VALUES:")
        print(pd.DataFrame.from_dict(pvalues_in, orient='index', columns = ['pvalues']))

        return in_model


    def stepwise_fwd_selection2(self, X, y, starting_features):

        # FIRST VARIABLE 
        # _____________________________________________________________________
        # model with first variable chosen from univariable analysis
        base_var = starting_features
        result = self.logit_on_subset(y=y, X=X, feature_set=base_var)
        base_loglik = result['loglik']
        loglik_ratio = 2 * (base_loglik - result['L0'])

        # create a dictionary of pvalues for variables added to the model
        pvalues_in = {}
        for var in base_var:
            pvalues_in[var] = result['result'].pvalues.loc[var]

        # SUBSEQUENT VARIABLES
        # _____________________________________________________________________
        i = 1 # keep track of iteration
        go = True # stopping rule

        in_model = base_var

        while go:

            # forward step --------------------------------------------
            pvalues_fwd = {}
            for var in self.variables:
                if var not in in_model:
                    variables = in_model + [var]
                    result = self.logit_on_subset(y=y, X=X, feature_set=variables)
                    loglik_ratio = 2 * (result['loglik'] - base_loglik)
                    pvalues_fwd[var] = chi2.sf(loglik_ratio, 1)
            
            # variable with smallest pvalue 
            min_var = min(pvalues_fwd, key=lambda k: pvalues_fwd[k])
            pvalue_min = pvalues_fwd[min_var]

            # add to the model only if it respects the threshold else, model is complete
            if pvalue_min < self.pvalue_e:
                in_model.append(min_var)
                pvalues_in[min_var] = pvalue_min
            else: 
                go = False

            # new model before backward check
            result = self.logit_on_subset(y=y, X=X, feature_set=in_model)
            base_loglik = result['loglik']
            print("NEW MODEL (FORWARD):")
            print(result['result'].summary2())
            # ---------------------------------------------------------

            # backward check ------------------------------------------
            pvalues = result['result'].pvalues
            pvalues_out = pvalues.where(pvalues > self.pvalue_z)
            vars_to_remove = pvalues_out.index.tolist()
            in_model = [x for x in in_model if x is not vars_to_remove]

            # new model after backward check
            result = self.logit_on_subset(y=y, X=X, feature_set=in_model)
            base_loglik = result['loglik']
            print("NEW MODEL (BACKWARD):")
            print(result['result'].summary2())
            # ----------------------------------------------------------

            # end of loop
            print("ROUND: "+str(i))
            i = i+1


        print("FINAL VARIABLES:")
        print(in_model)
        print("FINAL VARIABLES P-VALUES:")
        print(pd.DataFrame.from_dict(pvalues_in, orient='index', columns = ['pvalues']))

        return in_model

    

    def logit_on_subset(self, y, X, feature_set):
        # Fit model on feature_set and calculate RSS
        model = sm.Logit(y, X[feature_set])
        result = model.fit(maxiter=self.maxiter, method=self.solver)
        loglik = result.llf
        llr_pvalue = result.llr_pvalue

        return {"features":feature_set, "loglik":loglik, "pvalue": llr_pvalue, "result": result, "L0":result.llnull}
        

    def random_forest_on_subset(self, y, X, feature_set):
        model = RandomForestClassifier(n_estimators=100)
        result = model.fit(X[feature_set], y)

        return {"features":feature_set, "result": result}


if __name__ == '__main__':

    if not os.path.exists('logs'):
        os.makedirs('logs')

    y = pd.read_csv('preprocessed_data/accidents_train.csv', sep=',').ix[:,0]

    if config['model'] is "A1":
        X = pd.read_csv('preprocessed_data/data_train_initial.csv', sep=',')

        # identify and remove irrelevant variables 
        ua = UnivariableAnalysis(config, X, y)
        vars_to_remove = ua.vars_to_remove
        X.drop(vars_to_remove, axis=1, inplace=True)

        # first var for logit builder
        starting_features = ['const']
        first_var = ua.pvalues['pvalues'].idxmin()
        starting_features.append(str(first_var))

        # logit builder
        ma = MultivariableAnalysis(config)
        model_variables_A1 = ma.stepwise_fwd_selection2(X, y, starting_features)

        # logit model 
        logit_A1 = ma.logit_on_subset(y, X, model_variables_A1)
        # RF model 
        ranfor_A1 = ma.random_forest_on_subset(y, X, model_variables_A1)

        # save the logit model
        pickle.dump(logit_A1, open('logs/model_logit_A1.sav', 'wb'))

        # save the RF model
        pickle.dump(ranfor_A1, open('logs/model_ranfor_A1.sav', 'wb'))

        # save model variables
        np.savetxt("logs/model_variables_A1.csv", model_variables_A1, delimiter=",", fmt='%s')

    elif config['model'] is "A2":
        X = pd.read_csv('preprocessed_data/data_train_extended.csv', sep=',')

        # identify and remove irrelevant variables 
        ua = UnivariableAnalysis(config, X, y)
        vars_to_remove = ua.vars_to_remove
        X.drop(vars_to_remove, axis=1, inplace=True)

        # first var for logit builder
        starting_features = ['const']
        first_var = ua.pvalues['pvalues'].idxmin()
        starting_features.append(str(first_var))

        # logit builder
        ma = MultivariableAnalysis(config)
        model_variables_A2 = ma.stepwise_fwd_selection2(X, y, starting_features)

        # logit model 
        logit_A2 = ma.logit_on_subset(y, X, model_variables_A2)
        # RF model 
        ranfor_A2 = ma.random_forest_on_subset(y, X, model_variables_A2)

       # save the logit model 
        pickle.dump(logit_A2, open('logs/model_logit_A2.sav', 'wb'))

        # save the RF model 
        pickle.dump(ranfor_A2, open('logs/model_ranfor_A2.sav', 'wb'))

        # save model variables
        np.savetxt("logs/model_variables_A2.csv", model_variables_A2, delimiter=",", fmt='%s')

    elif config['model'] is "A3":
        X = pd.read_csv('preprocessed_data/data_train_extended.csv', sep=',')

        # Load variables from initial database (A1)
        starting_features = list(pd.read_csv('logs/model_variables_A1.csv', sep=',').ix[:,0])

        # perform stepwise selection in addition to starting features
        ma = MultivariableAnalysis(config)
        model_variables_A3 = pd.DataFrame(ma.stepwise_fwd_selection2(X_train, y_train, starting_features))

        model_variables_A3.to_csv("logs/model_variables_A3.csv", index=False, sep=",")

        # logit model 
        logit_A3 = ma.logit_on_subset(y, X, list(model_variables_A3))
        # RF model 
        ranfor_A3 = ma.random_forest_on_subset(y, X, list(model_variables_A3))

       # save the logit model 
        pickle.dump(logit_A3, open('logs/model_logit_A3.sav', 'wb'))

        # save the RF model 
        pickle.dump(ranfor_A3, open('logs/model_ranfor_A3.sav', 'wb'))



