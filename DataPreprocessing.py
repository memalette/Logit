import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import seaborn as sns
import itertools
import os

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer

from config_logit import config


class DataLoader():

    def __init__(self, config):

        
        self.accident_data = pd.read_csv("data/accident_data.csv")

        if config["database"] is "initial":
            data = pd.read_csv("data/initial_data.csv")


        if config["database"] is "extended":
            data = pd.read_csv("data/extended_data3.csv")

        # remove fleet id and nb_accicents
        self.nb_accidents = data.loc[:,'nb_accidents']
        data = data.drop(columns=['fleet_id', 'nb_accidents'])

        # group by data types
        dtype_dict = data.columns.to_series().groupby(data.dtypes).groups
        dtype_dict = {k.name: v for k, v in dtype_dict.items()}

        new_names = {'int64': 'disc', 'float64':'cont', 'object':'cat'}
        dtype_dict = {new_names[key]:list(value) for key, value in dtype_dict.items()}
            
        self.cat_data = data.loc[:,dtype_dict['cat']]
        self.disc_data = data.loc[:,dtype_dict['disc']]
        self.cont_data = data.loc[:,dtype_dict['cont']]
            

class PrimaryDataAnalysis():
    def __init__(self, config):
        dl = DataLoader(config)
        self.nb_accidents = dl.nb_accidents
        self.accident_data = dl.accident_data
        self.cat_data = dl.cat_data
        self.disc_data = dl.disc_data
        self.cont_data = dl.cont_data

        self.get_categories()

        self.get_variables_to_collapse()


    # This function returns a dataframe with the number of possible values discrete and categorical variables can take 
    def get_categories(self):
        cat_disc_data  = pd.concat([self.cat_data, self.disc_data], axis=1)
        var_names = cat_disc_data.columns.values
        categories = {}
        for var in var_names:
            categories[var] = len(cat_disc_data[var].unique())

        self.categories = pd.DataFrame.from_dict(categories, orient = 'index')
        print(self.categories)


    def frequency_table(self, var):
        accident_var = pd.crosstab(index=self.data_['x'], 
                           columns=self.data_[var], margins=True)

        accident_var = accident_var.rename(columns = {'All':'coltotal'})
        accident_var = accident_var.rename(index = {'All':'rowtotal'})
        frequency_table = accident_var/accident_var.ix["rowtotal"]

        print(frequency_table.round(2))

        # based on frequency table, should categories be collapsed?
        # can we find zero cells in the frequency table
        contains_zeros = (frequency_table == 0).any(axis=0)
        cat_with_zeros = contains_zeros.index[:-1]
        # convert to list
        contains_zeros = list(contains_zeros)[:-1]
        # get indices
        idx = [i for i, x in enumerate(contains_zeros) if x]
        # if it doesn contain zeros, it should collapse
        should_collapse = (len(idx) != 0)
        
        if should_collapse:
            # get category index to collapse with
            prev_idx = idx[0] - 1
            # and concatenate with idx 
            collapse_idx = [prev_idx] + idx
            # get the categories names instead of categoris indices
            collapse_cat = list(np.array(cat_with_zeros[np.array(collapse_idx)]))
        else:
            collapse_cat = None
      
        return should_collapse, collapse_cat


    def get_variables_to_collapse(self):
        # concat for frequency tables
        df_list = [self.accident_data, self.cat_data, self.disc_data]
        self.data_ = pd.concat(df_list, axis=1)

        # frequency table
        cat_disc_var = list(self.data_.columns.values)[1:]
        self.var_to_collapse = {}
        for var in cat_disc_var:
            should_collapse, collapse_cat = self.frequency_table(var)
            if should_collapse:
                self.var_to_collapse[var] = collapse_cat

        print(self.var_to_collapse)


class DataPreprocessing(DataLoader):

    def __init__(self, config):
        # configurations
        self.scale_cont = config["scale_cont"]
        self.scaler_type_cont = config["scaler_type_cont"]

        self.onehot_cat = config["onehot_cat"]

        self.treat_disc_as = config["treat_disc_as"]
        self.scale_disc = config["scale_disc"]
        self.scaler_type_disc = config["scaler_type_disc"]
        self.thermometer_disc = config["thermometer_disc"]

        self.test_percent = config["test_percent"]


        # Primary data analysis
        pda = PrimaryDataAnalysis(config)

        # save necessary data sets and information
        self.nb_accidents = pda.nb_accidents
        self.accident_data = pda.accident_data
        self.cat_data = pda.cat_data
        self.disc_data = pda.disc_data
        self.cont_data = pda.cont_data
        self.var_to_collapse = pda.var_to_collapse
        self.data_ = pda.data_ # data excluding continuous variables

        # Collapsing some of the variables
        self.collapse_categories()
        print("COLLAPSED")
        print(self.get_categories(self.data_))

        # get frequency tables after collapsing
        cat_disc_var = list(self.data_.columns.values)[1:]
        for var in cat_disc_var:
            self.frequency_table(var)

        # preprocess data
        self.preprocess_continuous()
        self.preprocess_categorical()
        self.preprocess_discrete()

        # Concatenate all data
        df_list = [self.cat_data, self.disc_data, self.cont_data]
        self.data = pd.concat(df_list, axis=1)
        self.hdr_data = list(self.data.columns.values)
        self.data = sm.add_constant(self.data)
        print(self.data.head())

        # label encode y
        self.X = self.data.values
        y = self.accident_data.values.reshape((self.X.shape[0], ))
        encoder = LabelEncoder()
        encoder.fit(y)
        self.y = encoder.transform(y)
        
        print(self.get_categories(self.data_))




    # This function returns a dataframe with the number of possible values discrete and categorical variables can take 
    def get_categories(self, data):
        var_names = data.columns.values
        categories = {}
        for var in var_names:
            categories[var] = len(data[var].unique())

        return pd.DataFrame.from_dict(categories, orient = 'index')
        


    # This function collapses the categories provided by the dictionary var_to_collapse for each varaibles in var_to_collapse
    def collapse_categories(self):
        for k in self.var_to_collapse.keys():
            new_cat = self.var_to_collapse[k][0]
            for i in self.var_to_collapse[k]:
                self.data_.loc[self.data_[k] == i,k] = new_cat


    def frequency_table(self, var):
        accident_var = pd.crosstab(index=self.data_['x'], 
                           columns=self.data_[var], margins=True)

        accident_var = accident_var.rename(columns = {'All':'coltotal'})
        accident_var = accident_var.rename(index = {'All':'rowtotal'})
        frequency_table = accident_var/accident_var.ix["rowtotal"]

        print(frequency_table.round(2))


    def preprocess_continuous(self):
        if self.scale_cont:
            self.cont_data = self.scale_data(scaler_ty=self.scaler_type_cont, data=self.cont_data)
        else: 
            pass

    def preprocess_categorical(self):
        if self.onehot_cat:
            self.cat_data = self.onehot_encode(self.cat_data)
        else: 
            pass

    def preprocess_discrete(self):
        if self.treat_disc_as is 'cont':
            if self.scale_disc:
                self.disc_data = self.scale_data(scaler_ty=self.scaler_type_disc, data=self.disc_data)
        elif self.treat_disc_as is 'cat':
            self.disc_data = self.onehot_encode(data=self.disc_data)
        elif self.treat_disc_as is 'disc':
            if self.thermometer_disc:
                self.disc_data = self.thermometer_encode(data=self.disc_data)
        else: 
            pass

    # this function scales the given data according to a specified scaler type
    def scale_data(self, scaler_ty, data):
        var_names = list(data.columns.values)

        if(scaler_ty is 'standard'):
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
        elif(scaler_ty is 'minmax'):
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
        elif(scaler_ty is 'robust'):
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(data)
        elif(scaler_ty is 'normalizer'):
            scaler = Normalizer()
            scaled_data = scaler.fit_transform(data)

        data = pd.DataFrame(scaled_data, columns=var_names)

        return data

    def onehot_encode(self, data):
        var_names = list(data.columns.values)
        oh_data = self.accident_data
        for var in var_names:
            oh_data_ = pd.get_dummies(data[var], prefix=var)
            oh_data  = pd.concat([oh_data, oh_data_], axis=1)
        
        data = oh_data.drop(oh_data.columns[0], axis=1)

        return data

    def thermometer_encode(self, data):
        var_names = list(data.columns.values)
        therm_data = self.accident_data
        for var in var_names:
            n = max(data[var])
            col_names = [var+'='+str(x) for x in range(1, n + 1)]
            encoding_mat = np.tril(np.ones((n, n), dtype=int))
            encoding_mat = np.concatenate((np.zeros((1,n), dtype=int), encoding_mat), axis=0)
            therm_data_ = pd.DataFrame(encoding_mat[data[var]], columns = col_names)
            therm_data  = pd.concat([therm_data, therm_data_], axis=1)

        data = therm_data.drop(therm_data.columns[0], axis=1)
        
        return data


if __name__ == '__main__':

    # Preprocess initial database
    config["database"] = "initial"
    dp1 = DataPreprocessing(config)

     # Preprocess extended database
     config["database"] = "extended"
     dp2 = DataPreprocessing(config)

     # Accident data will not change from one to another
     bin_disc_accidents = pd.concat([pd.DataFrame(dp1.y), dp1.nb_accidents], axis=1)

     # Train, validation, test indices
     indices = list(range(len(dp1.y)))
     i_train_val, i_test = train_test_split(indices, test_size=config['test_percent'])
     i_train, i_val = train_test_split(i_train_val, test_size=config['conditional_val_percent'])

     # Split the data
     # Accident data
     accidents_train = bin_disc_accidents.ix[i_train, :]
     accidents_val = bin_disc_accidents.ix[i_val, :]
     accidents_test = bin_disc_accidents.ix[i_test, :]
     # Initial db
     data_train1 = dp1.data.ix[i_train,:]
     data_val1 = dp1.data.ix[i_val,:]
     data_test1 = dp1.data.ix[i_test,:]
     # Extended db
     data_train2 = dp2.data.ix[i_train,:]
     data_val2 = dp2.data.ix[i_val,:]
     data_test2 = dp2.data.ix[i_test,:]

     # Save data
     if not os.path.exists('preprocessed_data'):
         os.makedirs('preprocessed_data')

     # Save accident data
     accidents_train.to_csv("preprocessed_data/accidents_train.csv", index=False, sep=",")
     accidents_val.to_csv("preprocessed_data/accidents_val.csv", index=False, sep=",")
     accidents_test.to_csv("preprocessed_data/accidents_test.csv", index=False, sep=",")

     # Save initial db
     data_train1.to_csv("preprocessed_data/data_train_initial.csv", index=False, sep=",")
     data_val1.to_csv("preprocessed_data/data_val_initial.csv", index=False, sep=",")
     data_test1.to_csv("preprocessed_data/data_test_initial.csv", index=False, sep=",")

     # Save extended db
     data_train2.to_csv("preprocessed_data/data_train_extended.csv", index=False, sep=",")
     data_val2.to_csv("preprocessed_data/data_val_extended.csv", index=False, sep=",")
     data_test2.to_csv("preprocessed_data/data_test_extended.csv", index=False, sep=",")






