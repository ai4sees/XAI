import argparse
from src import load_dataset, get_synthetic_data
from preprocess import preprocessor



parser = argparse.ArgumentParser()

parser.add_argument("dataset", type = str, default = "synthetic")
parser.add_argument("url", type = str, default = None)
parser.add_argument("scalar", type = str, default = "normalize")
parser.add_argument("window_size", type = int, default = 24)
args = parser.parse_args()

df = load_dataset(dataset = args.dataset, url = args.url)
data_processor = preprocessor(df)
