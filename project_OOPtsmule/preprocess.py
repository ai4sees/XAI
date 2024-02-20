import pandas as pd
import numpy as np
import dill




class preprocessor():
    def __init__(self, df, y):
        self.df = df.drop(y, axis=1)
        self.y = df[y]

    def clean_data(self, col, time_col=['index', 'No', 'day', 'month', 'year', 'hour']):
        for i in col:
            if i in self.df.index:
                self.df.drop(i, axis=1, inplace=True)
            else:
                print(f"{i} not in dataframe columns")


        for i in time_col:
            if i in self.df.index:
                self.df.drop(i, axis=1, inplace=True)
            else:
                print(f"{i} not in dataframe columns")


    def normalize(self):
        min_max_dict = {}
        for feature_name in self.df.columns:
            max_value = self.df[feature_name].max()
            min_value = self.df[feature_name].min()

            self.df[feature_name] = (self.df[feature_name] - min_value) / (max_value - min_value)

            min_max_dict[feature_name] = {'max': max_value, 'min': min_value}
            return min_max_dict

    def standardize(self):
        std_mean_dict = {}
        for feature_name in self.df.columns:
            std_value = self.df[feature_name].std()
            mean_value = self.df[feature_name].mean()

            self.df[feature_name] = (self.df[feature_name] - mean_value) / std_value

            std_mean_dict[feature_name] = {'std': std_value, 'mean': mean_value}
            return std_mean_dict



    def create_sequences(self, split_feature = None, n_steps = 24):
        dataset_X = []
        dataset_y = []

        X_temp = []
        if split_feature:
            for val in self.df[split_feature].unique():
                X_temp.append(self.df[:][self.df[split_feature] == val])
        else:
            X_temp.append(self.df)

        for x in X_temp:
            for i in range(len(x) - (n_steps + 2)):
                dataset_X.append(x.iloc[i:i + n_steps].values)
                dataset_y.append(self.y.iloc[i + n_steps + 1])

        return np.asarray(dataset_X), np.asarray(dataset_y)


    def get_df(self):
        return self.df

    def set_df(self, df):
        self.df = df

    def save_df(self):
        with open("data.dill", 'wb') as f:
            dill.dump(self.df, f)