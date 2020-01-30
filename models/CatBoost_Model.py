from models.BaseModel import Base_Model
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import logging
logging.basicConfig(filename='logger.log', format='%(asctime)s :%(levelname)s :%(message)s ', level=logging.INFO)

class CatBoost_Model(Base_Model):
    
    def plot_feature_importance(self):
        imp_df = pd.DataFrame()
        imp_df['feature'] = self.features
        imp_df['gain'] = self.model.feature_importances_
        plt.figure(figsize=(8, 8))
        imp_df = imp_df.sort_values(by = 'gain', ascending = False)
        sns.barplot(y = 'feature', x = 'gain', data= imp_df)
        plt.tight_layout()
        plt.show()
        
    def loss(self, y_true, y_pred):
        y_diff = abs(y_true - y_pred)
        mask = y_true != 0
        return np.mean(y_diff[mask]/y_true[mask])
    
    def train_model(self, train_set, val_set = None):
        verbosity = 100 if self.verbose else 0
        clf = CatBoostRegressor(**self.params)
        if val_set:
            return clf.fit(train_set['X'], 
                           train_set['y'], 
                           eval_set=(val_set['X'], val_set['y']),
                           verbose=verbosity, 
                           cat_features=self.categoricals)
        else:
            return clf.fit(train_set['X'], 
                           train_set['y'], 
                           verbose=verbosity, 
                           cat_features=self.categoricals)
    
    def convert_dataset(self, x_train, y_train):
        train_set = {'X': x_train, 'y': y_train}
        return train_set
    
    def get_params(self):
        return self.params
    
    def fit(self):
        y_pred = np.zeros((len(self.test_df), ))
        x_train, x_val = self.train_df[self.features], self.val_df[self.features]
        y_train, y_val = self.train_df[self.target], self.val_df[self.target]
        train_set, val_set = self.convert_dataset(x_train, y_train), self.convert_dataset(x_val, y_val)
        model = self.train_model(train_set, val_set)
        conv_x_val = self.convert_x(x_val)
        val_preds = model.predict(conv_x_val)
        self.val_df['predicted_case_count'] = val_preds
        x_temp_val = self.val_df.groupby('application_date').agg({self.target : 'sum',
                                                                        'predicted_case_count' : 'sum'})
        
        logging.info('MAPE: {}'.format(self.loss(x_temp_val[self.target].values
                                           , x_temp_val['predicted_case_count'].values)))
        
        x = pd.concat([x_train, x_val])
        y = pd.concat([y_train, y_val])
        
        train_set = self.convert_dataset(x, y)
        model = self.train_model(train_set)
        x_test = self.convert_x(self.test_df[self.features])
        preds = model.predict(x_test)
        
        return preds, x_temp_val['predicted_case_count'].values, model