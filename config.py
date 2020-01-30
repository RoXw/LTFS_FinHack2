TRAIN_FILE = '../input/train_fwYjLYX.csv' # location of train file
TEST_FILE = '../input/test_1eLl9Yf.csv' # location of test file
SUBMISSION_PATH = '../submission' # location to store submission file

HOLIDAYS = '../input/holidays.pkl' # External holidays from https://www.india.gov.in/calendar

LGB_PARAMS_SEGMENT1 = {'n_estimators': 15000,
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'mape',
                    'subsample': 0.8,
                    'subsample_freq': 1,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.8,
                    'max_depth': 4,
                    'num_leaves': 10,
                    'lambda_l1': 10,  
                    'lambda_l2': 10,
                    'early_stopping_rounds': 200,
                    'verbose': 1,
                   'random_seed': 42
                    }

FEATS_SEGMENT1 = ['case_count_last_year', 'case_count_last_quarter', 'is_month_end', 'dow','case_count_last_half_year', 'case_count_last_9'
                    , 'case_count_ma_year', 'case_count_ma_quarter', 'case_count_half_year', 'case_count_nine_months',
                    'case_count_median_year', 'case_count_median_quarter', 'case_count_median_half_year', 'Month', 'branch_id'
                    , 'case_count_std_year', 'holidays',  'case_count_next_month_mean', 'case_count_next_month_sum', 
                    'case_count_prev_week_sum', 'daysindifference']


CAT_PARAMS_SEGMENT1 = {'loss_function': 'RMSE',
                   'task_type': "CPU",
                   'iterations': 400,
                   'od_type': "Iter",
                    'depth': 5,
                  'colsample_bylevel': 0.8, 
                   'early_stopping_rounds': 200,
                    'l2_leaf_reg': 10,
                   'random_seed': 42,
                    }


LGB_PARAMS_SEGMENT2 = {'n_estimators': 10000,
                    'boosting_type': 'gbdt',
                    'objective': 'poisson',
                    'metric': 'mape',
                    'subsample': 0.8,
                    'subsample_freq': 1,
                    'learning_rate': 0.03,
                    'feature_fraction': 0.8,
                    'max_depth': 6,
                    'num_leaves': 16,
                    'lambda_l1': 10,  
                    'lambda_l2': 10,
                    'early_stopping_rounds': 100,
                    'verbose': 1,
                    'random_seed': 42
                    }

FEATS_SEGMENT2 = ['case_count_last_year', 'case_count_last_quarter', 'is_month_end', 'dow', 'case_count_last_half_year',
                'case_count_last_9', 'case_count_ma_year', 'case_count_ma_quarter', 'case_count_half_year', 'case_count_nine_months',
                'case_count_median_year', 'case_count_median_quarter', 'case_count_median_half_year', 'Month', 'branch_id', 'day_of_year'
                , 'case_count_std_year', 'case_count_skew_year', 'dom', 'holidays', 'case_count_prev_month_mean', 'case_count_next_month_mean'
                , 'case_count_next_month_sum', 'daysindifference']


CAT_PARAMS_SEGMENT2 = {'loss_function': 'RMSE',
                   'task_type': "CPU",
                   'iterations': 400,
                   'od_type': "Iter",
                    'depth': 6,
                  'colsample_bylevel': 0.8, 
                   'early_stopping_rounds': 100,
                    'l2_leaf_reg': 1,
                   'random_seed': 42,
                    }

CAT_FEATS_SEGMENT2 = ['case_count_last_year', 'case_count_last_quarter', 'is_month_end', 'dow', 'case_count_last_half_year', 'case_count_last_9'
                        , 'case_count_ma_year', 'case_count_ma_quarter', 'case_count_half_year', 'case_count_nine_months',
                        'case_count_median_year', 'case_count_median_quarter', 'case_count_median_half_year', 'Month', 'branch_id', 'day_of_year'
                        , 'case_count_std_year', 'case_count_skew_year', 'dom', 'predicted_trend', 'case_count_next_month_sum', 'daysindifference']