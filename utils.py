import config
from models.LinearRegression import LinearRegression

import os
import pandas as pd
import numpy as np
import datetime
import pickle
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import multiprocessing 


def MAPE(y_true, y_pred):
    """
    y_true: numpy array
    y_pred: numpy array
    """
    y_diff = abs(y_true - y_pred)
    mask = y_true != 0
    return 100*np.mean(y_diff[mask]/y_true[mask])


def create_test_dataset(test, train, segment = 1):
    """
    Function to expand test dataset to include all branch_id's for all application dates
    test: pandas dataframe
    branch_id: numpy array
    """
    if segment == 1:
        branch_id = train['branch_id'].unique()
        branch_state = train.groupby('branch_id')['state'].apply(lambda x: np.unique(x)[0])
        branch_zone = train.groupby('branch_id')['zone'].apply(lambda x: np.unique(x)[0])
        N = len(test)
        test = pd.concat([test]*len(branch_id))
        test['branch_id'] = list(branch_id) * N
        test['state'] = test['branch_id'].map(branch_state)
        test['zone'] = test['branch_id'].map(branch_zone)
        return test
    else:
        state = train['state'].unique()
        state_zone = train.groupby('state')['zone'].apply(lambda x: np.unique(x)[0])
        N = len(test)
        test = pd.concat([test]*len(state))
        test['state'] = list(state) * N
        test['zone'] = test['state'].map(state_zone)
        return test

    
def load_data(segment=1):
    '''
    Load data from local for a given segment and splits the data into train and val
    For validation, selected last 3 months data
    Args:
        segment: 1/2 indicating the segment for which to load data
    Returns:
        train: Train pandas dataframe
        test: test pandas dataframe
        val: validation pandas dataframe
    '''
    train = pd.read_csv(config.TRAIN_FILE)
    test = pd.read_csv(config.TEST_FILE)
    
    train = train.loc[train.segment == segment].reset_index(drop = True)
    test = test.loc[test.segment == segment].reset_index(drop = True)
    
    train['is_train'] = True
    test['is_train'] = False
    
    train['application_date'] = pd.to_datetime(train['application_date'], yearfirst=True)
    test['application_date'] = pd.to_datetime(test['application_date'], yearfirst=True)
    
    train['application_year'] = train['application_date'].dt.year
    test['application_year'] = test['application_date'].dt.year
    
    train = train.sort_values(by = ['application_date', 'segment', 'state'])
    test = test.sort_values(by = ['application_date', 'segment'])
    
    train['Month'] = train['application_date'].dt.month
    train['day_of_year'] = train['application_date'].dt.dayofyear
    
    ## removed branches having 0's case_counts for all the application dates
    if segment == 1:
        remove_branch_ids = {263, 265, 267, 271, 264, 266, 262, 268, 269, 270}
    else:
        remove_branch_ids = {4, 11}
    
    test['Month'] = test['application_date'].dt.month
    test['day_of_year'] = test['application_date'].dt.dayofyear
    
    val = train.loc[train['application_date'] >= 
                    train['application_date'].max() - datetime.timedelta(test['application_date'].nunique())].reset_index(drop=True)
    
    train = train.loc[train['application_date'] <
                    train['application_date'].max() - datetime.timedelta(test['application_date'].nunique())].reset_index(drop=True)
    
    test.drop(columns = ['id'], inplace = True)
    
    test = create_test_dataset(test, train, segment=segment)
    
    if segment == 2:
        def lb_enc(train, test, val):
            lb_dict = {}
            for v in train["state"].unique():
                if v not in lb_dict:
                    lb_dict[v] = len(lb_dict)
                    
            for v in test["state"].unique():
                if v not in lb_dict:
                    lb_dict[v] = len(lb_dict)
                    
            for v in val["state"].unique():
                if v not in lb_dict:
                    lb_dict[v] = len(lb_dict)
            

            train["branch_id"] = train["state"].map(lb_dict)
            test["branch_id"] = test["state"].map(lb_dict)
            val["branch_id"] = val["state"].map(lb_dict)
        
        lb_enc(train, test, val)
    train = train.loc[~train['branch_id'].isin(remove_branch_ids)].reset_index(drop=True)
    
    return train, val, test


def create_features_group_level(group):
    """
    create features at group level, for segment 1: group is branch_id
    for segment 2: group is state 
    Args:
        group: dataframe containing a single branch_id/state
    Returns:
        dataframe with feature values
    """

    def get_values(years, months):
        """
        Function to get past case count value given year and month
        Args:
            years: Number of years in the past
            months: Number of months in the past
        Returns:
            value if for that application date case count exists else np.nan
        """
        return [group.loc[edt - pd.tseries.offsets.DateOffset(years = years, months = months), 'case_count'] 
                if edt - pd.tseries.offsets.DateOffset(years = years, months = months) in group.index 
                else np.nan for edt in group.index
                ]
    
    def get_summary(years, months, func):
        """
        Summary of case_count for the past years and month
        Args:
            years: Number of years in the past
            month: Number of months in the past
        """
        values = [func(group.loc[edt - pd.tseries.offsets.DateOffset(years = years
                                                            , months = months):edt
                        , 'case_count'])
                  for edt in group.index
                ]
        return values
    
    def get_prev_summary(func, flag):
        """
        Function to get prev/next year, month, week summary for the past year
        Args:
            func: function to summarize
            flag: nextmonth: to get summarized case count for the next month in previous year
                  prevmonth: to get summarized case count for the prev month in previous year
                  week: to get summarized case count for the next week in previous year
        Returns:
            summarized value for the appropriate flag
        """
        if flag == 'nextmonth':
            values = [func(group.loc[edt - \
                                     pd.tseries.offsets.DateOffset(years = 1):edt - \
                                     pd.tseries.offsets.DateOffset(months = 11)
                        , 'case_count'])
                  for edt in group.index
                ]
            return values
        elif flag == 'prevmonth':
            values = [func(group.loc[edt - \
                                     pd.tseries.offsets.DateOffset(months = 13):edt - \
                                     pd.tseries.offsets.DateOffset(years = 1)
                        , 'case_count'])
                  for edt in group.index
                ]
            return values
        elif flag == 'week':
            values = [func(group.loc[edt - \
                                     pd.tseries.offsets.DateOffset(years = 1):edt - \
                                     pd.tseries.offsets.DateOffset(weeks = 51)
                        , 'case_count'])
                  for edt in group.index
                ]
            return values
        else:
            raise Exception("Invalid flag")

    lr = LinearRegression(group.loc[group['is_train']].reset_index(drop=True), 
                             group.loc[~group['is_train']].reset_index(drop=True),
                             val_df = None, features = None, categoricals=[]
                              , params_dict = {}, verbose=True)
        
    group.loc[group['is_train'], 'predicted_trend'] = lr.oof_pred
    group.loc[~group['is_train'], 'predicted_trend'] = lr.y_pred
    tmep = group.loc[group.case_count > 0, 'application_date'].min()
    group['start_date'] = [tmep] * len(group)
    group['daysindifference'] = group['application_date'] - group['start_date']
    group['daysindifference'] = group['daysindifference'].astype(int).apply(lambda x: x / (24 * 3600 * 10e8))
    group['daysindifference'] = np.clip(group['daysindifference'], 0, group['daysindifference'].max())
    
    group['dow'] = group['application_date'].dt.dayofweek
    group['dom'] = group['application_date'].dt.day
    group['is_month_end'] = group['application_date'].dt.is_month_end
    group['is_month_begin'] = group['application_date'].dt.is_month_start
    group['is_quarter_end'] = group['application_date'].dt.is_quarter_end
    group['is_quarter_start'] = group['application_date'].dt.is_quarter_start
    
    group = group.set_index('application_date')
    group['case_count_last_year'] = get_values(years = 1, months = 0)
            
    group['case_count_last_quarter'] = get_values(years = 0, months = 4)
    group['case_count_last_half_year'] = get_values(years = 0, months = 6)
    group['case_count_last_9'] = get_values(years = 0, months = 9)
    group['case_count_ma_year'] = get_summary(years = 1, months = 0, func=np.nanmean)
    group['case_count_skew_year'] = get_summary(years = 1, months = 0, func=skew)
    group['case_count_ma_quarter'] = get_summary(years = 0, months = 4, func=np.nanmean)
    group['case_count_half_year'] = get_summary(years = 0, months = 6, func=np.nanmean)
    group['case_count_nine_months'] = get_summary(years = 0, months = 9, func=np.nanmean)
    group['case_count_median_year'] = get_summary(years = 1, months = 0, func=np.nanmedian)
    group['case_count_median_quarter'] = get_summary(years = 0, months = 4, func=np.nanmedian)
    group['case_count_median_half_year'] = get_summary(years = 0, months = 6, func=np.nanmedian)
    group['case_count_median_nine_months'] = get_summary(years = 0, months = 9, func=np.nanmedian)
    
    group['case_count_std_year'] = get_summary(years = 1, months = 0, func=np.nanstd)
    group['case_count_std_quarter'] = get_summary(years = 0, months = 4, func=np.nanstd)
    
    group['case_count_prev_month_mean'] = get_prev_summary(np.mean, flag='prevmonth')
    group['case_count_next_month_mean'] = get_prev_summary(np.mean, flag='nextmonth')
    group['case_count_prev_week_mean'] = get_prev_summary(np.mean, flag='week')
    
    group['case_count_prev_month_sum'] = get_prev_summary(np.sum, flag='prevmonth')
    group['case_count_next_month_sum'] = get_prev_summary(np.sum, flag='nextmonth')
    group['case_count_prev_week_sum'] = get_prev_summary(np.sum, flag='week')        

    return group.reset_index()

def create_features(data, N):
    """
    Create all features required by the model at branch_id level
    Args:
        data : pandas dataframe
        N: int denoting the length of test data for each branch
    Returns:
        pandas dataframe containing all features
    """
    holidays = pickle.load(open(config.HOLIDAYS, "rb"))

    data['holidays'] = data['application_date'].astype(str).map(holidays)

    def lb_enc(data, col):
        lb_dict = dict()
        for v in data[col].unique():
            if v not in lb_dict:
                lb_dict[v] = len(lb_dict)
            
        data[col] = data[col].map(lb_dict)
    
    lb_enc(data, 'holidays')
    
    features = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    features = pool.map(create_features_group_level, (group for name, group in data.groupby('branch_id', sort=False)))

    return pd.concat(features, axis = 0).reset_index(drop=True)


def create_data(segment = 1, aggregate = False):
    '''
    helper function to call load and create features
    Arguments:
        segment: 1/2 int
        aggregate: True, if want to create the featureas and model at aggregate level
                    False, if want to create the features and model at branch_id, application date level
    Returns:
        train: Train pandas dataframe
        test: test pandas dataframe
        val: validation pandas dataframe
    '''
    train, val, test = load_data(segment=segment)
    data = pd.concat([train, val, test])
    
    if aggregate:
        data = data.groupby('application_date').agg({'Month': 'mean',
                                            'application_year': 'mean',
                                            'case_count': 'sum',
                                            'day_of_year': 'mean',
                                            'is_train': 'median',
                                            'branch_id': 'median'}).reset_index()


    data = data.sort_values(by = ['branch_id','application_date']).reset_index(drop=True)
    
    data_features = create_features(data, N = test['application_date'].nunique())
    
    data_features = data_features.reset_index()
    
    data_features['application_date'] = pd.to_datetime(data_features['application_date'], yearfirst=False)

    train = data_features.loc[data_features['is_train']].reset_index(drop=True)
    test = data_features.loc[~data_features['is_train']].reset_index(drop=True)
    
    val = train.loc[train['application_date'] >= 
                    train['application_date'].max() - datetime.timedelta(test['application_date'].nunique())].reset_index(drop=True)
    
    train = train.loc[train['application_date'] <
                    train['application_date'].max() - datetime.timedelta(test['application_date'].nunique())].reset_index(drop=True)

    if segment == 1:
        train = train.loc[train['case_count_last_year'].notna()].reset_index(drop=True)
        val = val.loc[val['application_date'] != '2019-07-05']
    else:
        train = train.loc[train['application_date'] > train['application_date'].max() - datetime.timedelta(365)].reset_index(drop=True)
    
    return train, val, test


def sub_creation(lgbm_models, cat_models, lgb_models_seg2, cat_models_seg2, test1, test, fileName):
    """
    Function to create submission file
    Arguments:
        lgbm_models: list of lightgbm models for segment 1
        cat_models: list of catboost models for segment 1
        lgb_models_seg2: list of lightgbm models for segment 2
        cat_models_seg2: list of catboost models for segment 2
        test1: test data for segment 1
        test: test data for segment 2
        fileName: String, name of the submission file to create
    """
    sub = pd.read_csv(config.TEST_FILE)
    
    sub['application_date'] = pd.to_datetime(sub['application_date'], yearfirst=True)

    test1['prediction'] = np.mean([model.y_pred for model in lgbm_models], axis =0) * 0.3 + \
                            np.mean([model.y_pred for model in cat_models], axis =0) * 0.7

    preds = test1.groupby('application_date')['prediction'].sum().reset_index()

    preds['application_date'] = pd.to_datetime(preds['application_date'], yearfirst=False)
    preds.columns = ['application_date', 'case_count']

    sub1 = sub.loc[:86].reset_index(drop = True)

    sub1 = sub1.join(preds.set_index('application_date'), on = 'application_date', how = 'left')

    sub1['case_count'] = np.clip(sub1['case_count'], sub1['case_count'].min(), 4500)

    test['prediction'] = np.mean([model.y_pred for model in lgb_models_seg2], axis =0) * 0.8 + \
                        np.mean([model.y_pred for model in cat_models_seg2], axis =0) * 0.2

    preds = test.groupby('application_date')['prediction'].sum().reset_index()
                        

    preds['application_date'] = pd.to_datetime(preds['application_date'], yearfirst=False)
    preds.columns = ['application_date', 'case_count']

    sub2 = sub.loc[87:].reset_index(drop = True)
    sub2 = sub2.join(preds.set_index('application_date'), on = 'application_date', how = 'left')

    sub = pd.concat([sub1, sub2], ignore_index=True)
    sub['case_count'] = sub['case_count'].astype(int)

    sub.to_csv(os.path.join(config.SUBMISSION_PATH, f'{fileName}.csv'), index=False)