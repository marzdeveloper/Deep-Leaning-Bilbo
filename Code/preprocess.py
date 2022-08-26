__author__ = 'Daniele Marzetti'
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
import numpy as np

path = "/content/drive/MyDrive/Cyber Security/"
nfolds = 10

def create_set_label():
  dataset = pd.read_csv(path + "dga_domains_full.csv", encoding= "utf-8", names=['label', 'family', 'domain'])
  y = pd.Series(dataset['label'] == 'dga', dtype=int)
  x = pd.Series(dataset['domain'])
  x.to_csv(path + "x.csv", header=False, index=False)
  y.to_csv(path + "y.csv", header=False, index=False)
  return x, y, x.map(len).max()


def conversion(x, mapping):
  converted = []
  for y in list(x):
    converted.append(mapping.get(y))
  return converted

def to_numeric(dataset, MAX_STRING_LENGTH):
  valid_characters = set()
  dataset_characters = dataset.map(list)
  [valid_characters.add(j) for i in dataset_characters for j in i]

  charachetrs_map = dict.fromkeys(valid_characters)

  for i in charachetrs_map.keys():  #ordinamento personalizzato
    if ord(i)>47 and ord(i)<58:
      charachetrs_map[i]=ord(i)-21
    if ord(i)>96:
      charachetrs_map[i]=ord(i)-96

  MAX_INDEX = max((filter(None.__ne__,list(charachetrs_map.values()))))

  for i in sorted(charachetrs_map.keys()):
    if charachetrs_map[i]==None:
      MAX_INDEX+=1
      charachetrs_map[i]=MAX_INDEX

  dataset_preprocess = dataset.apply(conversion, mapping = charachetrs_map)
  dataset_final = pad_sequences(dataset_preprocess.to_numpy(), maxlen=MAX_STRING_LENGTH, padding="pre", value=0)
  return  dataset_final, MAX_INDEX

def kfold(x,y):
  # Divide the dataset into training + holdout and testing with folds
  sss = StratifiedKFold(n_splits=nfolds)

  fold = 0
  for train, test in sss.split(x, y):
    print("Writing fold " + str(fold + 1) + " to csv...")
    fold += 1
    x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
    np.savetxt(path +"x_train" + str(fold) + ".csv", x_train, fmt='%i', delimiter=',')
    np.savetxt(path + "x_test" + str(fold) + ".csv", x_test, fmt='%i', delimiter=',')
    np.savetxt(path + "y_train" + str(fold) + ".csv", y_train, fmt='%i', delimiter=',')
    np.savetxt(path + "y_test" + str(fold) + ".csv", y_test, fmt='%i', delimiter=',')
  print("Files created")