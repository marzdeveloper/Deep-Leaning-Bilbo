__author__ = 'Daniele Marzetti'
from preprocess import *
from utils import *
from network import *

start_fold = 1
end_fold = 10

dataset, y, MAX_STRING_LENGTH = create_set_label()
x, MAX_INDEX = to_numeric(dataset, MAX_STRING_LENGTH)
kfold(x,y)
use_tpu()
train_eval_test(MAX_STRING_LENGTH, MAX_INDEX + 1, start_fold, end_fold)
metrics_avg()
