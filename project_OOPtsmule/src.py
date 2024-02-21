import numpy as np
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen


def get_synthetic_data():
    # Preparing lists to store the data
    a, b, c, d, e, f = [], [], [], [], [], []

    # First loop
    a.extend(np.sort(np.random.uniform(0, 0.5, 400)))
    b.extend(np.sort(np.random.uniform(0, 0.5, 400)))
    c.extend(np.random.uniform(0.5, 1, 400))
    d.extend(np.sort(np.random.uniform(0.5, 1, 400))[::-1])
    e.extend(np.random.uniform(0.5, 1, 400))
    f.extend(np.random.uniform(0.5, 1, 400))

    a.extend(np.sort(np.random.uniform(0, 0.5, 400))[::-1])
    b.extend(np.sort(np.random.uniform(0, 0.5, 400))[::-1])
    c.extend(np.random.uniform(0.5, 1, 400))
    d.extend(np.random.uniform(0.5, 1, 400))
    e.extend(np.sort(np.random.uniform(0.5, 1, 400)))
    f.extend(np.random.uniform(0.5, 1, 400))

    for i in range(400):
        a.append(0.66)
        b.append(0.457)
    c.extend(np.random.uniform(0.5, 1, 400))
    d.extend(np.random.uniform(0.5, 1, 400))
    e.extend(np.random.uniform(0.5, 1, 400))
    f.extend(np.random.uniform(0.5, 1, 400))

    a.extend(np.random.uniform(0, 0.5, 400))
    b.extend(np.random.uniform(0, 0.5, 400))
    c.extend(np.random.uniform(0.5, 1, 400))
    for i in range(400): d.append(0.33)
    e.extend(np.sort(np.random.uniform(0.5, 1, 400)))
    f.extend(np.sort(np.random.uniform(0.5, 1, 400)))

    a.extend(np.random.uniform(0.5, 1, 400))
    b.extend(np.random.uniform(0.5, 1, 400))
    c.extend(np.random.uniform(0, 0.5, 400))
    d.extend(np.random.uniform(0, 0.5, 400))
    e.extend(np.sort(np.random.uniform(0.5, 1, 400))[::-1])
    f.extend(np.sort(np.random.uniform(0.5, 1, 400))[::-1])

    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    d = np.asarray(d)
    e = np.asarray(e)
    f = np.asarray(f)

    g = (-44 * a - 32 * b + 0 * c + 8 * d + e ** 2 - f ** (1 / 2)) / 100

    # Creating DataFrame from the lists
    data = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'g': g})  # , 'e': e, 'f': f, 'g': g})

    return data


def load_dataset(dataset="beijing_pm2.5", url=None):
    if dataset == "beijing_pm2.5":
        dataset_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv'
        df = pd.read_csv(dataset_link)


    elif dataset == "synthetic_data":
        df = get_synthetic_data()

    elif dataset == "beijing_multi_site":
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

    else:
        dataset_link = url
        if ".csv" in dataset_link:
            df = pd.read_csv(dataset_link)

        if ".zip" in dataset_link:
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


def add_metadata(df, contribution_method, orig_score, pert_score,
                 rand_score, pert_score_metric):
    new_row = {
        "Contribution Method": contribution_method,
        "Original Score": orig_score,
        "Perturbation Score": pert_score,
        "Random Score": rand_score,
        "Perturbation Score Metric": pert_score_metric
    }

    # Append the new row to the DataFrame and return it
    return df.append(new_row, ignore_index=True)
