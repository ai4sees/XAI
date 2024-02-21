import argparse
from src import load_dataset, get_synthetic_data
from preprocess import Preprocessor
import dill

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="synthetic")
parser.add_argument("--url", type=str, default=None)
parser.add_argument("--y_column", type=str, default='g')
parser.add_argument("--drop_col", nargs='+', default=None)
parser.add_argument("--scaler", type=str, default="normalize")
parser.add_argument("--window_size", type=int, default=24)
args = parser.parse_args()

df = load_dataset(dataset=args.dataset, url=args.url)
data_processor = Preprocessor(df, args.y_column)
data_processor.clean_data(col=args.drop_col)
if args.scaler == "normalize":
    _ = data_processor.normalize()

if args.scaler == "standardize":
    _ = data_processor.standardize()

X, y = data_processor.create_sequences(n_steps=args.window_size)
print(X.shape)
print(y.shape)
data = (X, y)
df_ = data_processor.get_df()
print(df_.head())
with open("data.dill", "wb") as f:
    dill.dump(data, f)
