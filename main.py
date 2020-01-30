import config
from utils import create_data
from utils import sub_creation
from utils import MAPE

from models.LGBM_Model import LGBM_Model
from models.CatBoost_Model import CatBoost_Model

import numpy as np
import gc

import logging
logging.basicConfig(filename='logger.log', format='%(asctime)s :%(levelname)s :%(message)s ', level=logging.INFO)

def main():
    ######################################### Segment 1 modelling #######################################
    logging.info("Started Data processing for segment 1")
    train, val, test1 = create_data(segment = 1)
    logging.info("Done data processing for segment 1")

    logging.info("Lightgbm model for segment 1")
    lgbm_params = config.LGB_PARAMS_SEGMENT1
    lgbm_models = []
    for seed in [42, 101, 454]:
        lgbm_params['random_seed'] = seed
        lgbm_model = LGBM_Model(train, test1, val, config.FEATS_SEGMENT1, [2, 3] ,lgbm_params, False)
        lgbm_models.append(lgbm_model)

    logging.info("Done lightGBM model for segment 1")

    test1['branch_id'] = test1['branch_id'].astype(int)
    val['branch_id'] = val['branch_id'].astype(int)
    train['branch_id'] = train['branch_id'].astype(int)

    logging.info("Catboost model for segment 1")
    cat_models = []
    cat_params = config.CAT_PARAMS_SEGMENT1
    for seed in [42, 101, 454]:
        cat_params['random_seed'] = seed
        cat_model = CatBoost_Model(train, test1, val, config.FEATS_SEGMENT1 , [2, 3, 13, 14, 16] ,cat_params, False)
        cat_models.append(cat_model)

    logging.info("Done CatBoost model for segment 1")

    temp = val.groupby('application_date').agg({'case_count': 'sum'})
    temp['prediction'] = np.mean([model.oof_pred for model in lgbm_models], axis =0) * 0.25 + \
                            np.mean([model.oof_pred for model in cat_models], axis =0) * 0.75

    logging.info("3 month CV for segment 1: {}".format(MAPE(temp['case_count'], temp['prediction'])))

    ######################################### Segment 2 modelling #######################################
    logging.info("Started Data processing for segment 2")
    train, val, test = create_data(segment = 2)
    logging.info("Done data processing for segment 2")

    logging.info("Lightgbm model for segment 2")
    lgb_models_seg2 = []
    lgbm_params = config.LGB_PARAMS_SEGMENT2
    for seed in [42, 23, 45, 5456, 454]:
        lgbm_params['random_seed'] = seed
        lgbm_model2 = LGBM_Model(train, test, val, config.FEATS_SEGMENT2, [3, 4] ,lgbm_params, False)
        lgb_models_seg2.append(lgbm_model2)

    logging.info("Done Lightgbm model for segment 2")

    test['branch_id'] = test['branch_id'].astype(int)
    val['branch_id'] = val['branch_id'].astype(int)
    train['branch_id'] = train['branch_id'].astype(int)

    logging.info("Catboost model for segment 2")
    cat_models_seg2 = []
    cat_params = config.CAT_PARAMS_SEGMENT2
    for seed in [23, 45, 5456]:
        cat_params['random_seed'] = seed
        cat_model2 = CatBoost_Model(train, test, val, config.CAT_FEATS_SEGMENT2, [2, 3, 14, 15, 18] ,cat_params, verbose=False)
        cat_models_seg2.append(cat_model2)

    logging.info("Done CatBoost model for segment 2")

    temp = val.groupby('application_date').agg({'case_count': 'sum'})
    temp['prediction'] = np.mean([model.oof_pred for model in lgb_models_seg2], axis =0) * 0.8 + \
                            np.mean([model.oof_pred for model in cat_models_seg2], axis =0) * 0.2

    logging.info("3 month CV for segment 2: {}".format(MAPE(temp['case_count'], temp['prediction'])))

    logging.info("Creating submission")

    sub_creation(lgbm_models, cat_models, lgb_models_seg2, cat_models_seg2, test1, test, "submission")
    gc.collect()


if __name__ == '__main__':
    main()

    

    
    

    

    




