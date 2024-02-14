import sys
import pandas as pd
import numpy as np

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import dill
import argparse




def preprocess_data(df, normalize=True, standardize=False):
    df_temp = df.copy()
    df_temp = df_temp.dropna()
    df_temp = df_temp.reset_index()

    col = ['index', 'No', 'day', 'month', 'year', 'hour', 'PM10']
    for i in col:
        if i in df_temp.columns:
            df_temp.drop(i, axis=1, inplace = True)

    if "wd" in df_temp.columns:
        df_temp['wd'] = df_temp['wd'].astype('category').cat.codes

    if "cbwd" in df_temp.columns:
        df_temp['cbwd'] = df_temp['cbwd'].astype('category').cat.codes


    if "station" in df_temp.columns:
        df_temp['station'] = df_temp['station'].astype('category').cat.codes

    min_max_dict = {}
    std_mean_dict = {}

    if normalize:

        for feature_name in df_temp.columns:
            max_value = df_temp[feature_name].max()
            min_value = df_temp[feature_name].min()

            df_temp[feature_name] = (df_temp[feature_name] - min_value) / (max_value - min_value)

            min_max_dict[feature_name] = {'max': max_value, 'min': min_value}

    elif standardize:

        for feature_name in df_temp.columns:
            std_value = df_temp[feature_name].std()
            mean_value = df_temp[feature_name].mean()

            df_temp[feature_name] = (df_temp[feature_name] - mean_value) / std_value

            std_mean_dict[feature_name] = {'std': std_value, 'mean': mean_value}
            
    return df_temp, min_max_dict, std_mean_dict





def create_sequences(X, y, split_feature=None, n_steps=24):
    dataset_X = []
    dataset_y = []
    
    X_temp = []
    if split_feature:
        for val in X[split_feature].unique():
            X_temp.append(X[:][X[split_feature] == val])
    else:
        X_temp.append(X)

    for x in X_temp:
        for i in range(len(x) - (n_steps + 2)):
            dataset_X.append(x.iloc[i:i + n_steps].values)
            dataset_y.append(y.iloc[i + n_steps + 1])

    return np.asarray(dataset_X), np.asarray(dataset_y)






def load_dataset(dataset="beijing_pm2.5"):
    if dataset=="beijing_pm2.5":
        dataset_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv'
        df = pd.read_csv(dataset_link)

        
    elif dataset=="beijing_multi_site":
        dataset_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip'
        resp = urlopen(dataset_link)
        zipfile = ZipFile(BytesIO(resp.read()))
        file_list = zipfile.namelist()
        df_list = []

        for file in file_list:
            if 'csv' in file:
                df_ = pd.read_csv(zipfile.open(file))
                df_list.append(df_)

        df = df_list[0]

    return df


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("dataset", type = str, default = "synthetic")
  parser.add_argument("scalar", type = str, default = "normalize")
  parser.add_argument("window_size", type = int, default = 24)
  args = parser.parse_args()


  df = load_dataset(args.dataset)
  if args.scalar =="normalize":
        normalize = True
        standardize = False

  elif args.scalar == "standardize":
        normalize = False
        standardize = True

  df_temp, _, _ = preprocess_data(df, normalize, standardize)
  X = df_temp.drop(['pm2.5'], axis=1)
  y = df_temp['pm2.5']

  data = create_sequences(X, y, n_steps = args.window_size)


  with open("data.dill", 'wb') as f:
      dill.dump(data, f)



















