import pandas as pd
from sklearn.linear_model import SGDClassifier
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from joblib import dump


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--train-in", action="store", dest="train_in")
parser.add_argument("--sklearn-model-out", action="store", dest="sklearn_model_out")
args = parser.parse_args()

input_data = args.train_in
output_data = args.sklearn_model_out

data_train = pd.read_json(input_data, lines=True)
lc = SGDClassifier()
model = lc.fit(data_train.iloc[:,1:], data_train.iloc[:,0])
dump(model, "{}.joblib".format(output_data))