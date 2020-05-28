import nni
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

    # get parameters from tuner
    RECEIVED_PARAMS = nni.get_next_parameter()
    logger.info("Received params:\n", RECEIVED_PARAMS)
    
    # list is a column_name generate from tuner
    df = pd.read_csv(file_name)
    df['Class'] = LabelEncoder().fit_transform(df['Class'])
    if 'sample_feature' in RECEIVED_PARAMS.keys():
        sample_col = RECEIVED_PARAMS['sample_feature']
    else:
        sample_col = []
    
    # raw feaure + sample_feature
    df = name2feature(df, sample_col, target_name)
    feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)
    nni.report_final_result({
        "default":val_score, 
        "feature_importance":feature_imp
    })

