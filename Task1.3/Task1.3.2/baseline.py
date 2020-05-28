
import logging
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

from fe_util import *
from model import *

logger = logging.getLogger('auto-fe-examples')

if __name__ == '__main__':
    file_name = 'german.csv'
    target_name = 'Class'
    id_index = 'Id'

   
    # list is a column_name generate from tuner
    df = pd.read_csv(file_name)
    df['Class'] = LabelEncoder().fit_transform(df['Class'])
    

    feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)
    
    print("val_score:",val_score)
