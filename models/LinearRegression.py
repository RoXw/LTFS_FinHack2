from models.BaseModel import Base_Model
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer

import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

class LinearRegression(Base_Model):
    
    def convert_x(self, x_train):
        x_train = sm.add_constant(x_train.reshape((-1, 1)), prepend=False)
        return x_train
    
    def fit(self):
        folds = KFold(n_splits=10, random_state=11, shuffle=True)
        y = self.train_df[self.target]
        oof_preds = np.zeros((len(self.train_df)))
        qt = QuantileTransformer(output_distribution='normal', random_state=11)
        qt.fit(y.values.reshape((-1, 1)))
        test_preds = []
        y = qt.transform(y.values.reshape((-1, 1)))
        X = np.asarray(list(range(0, len(self.train_df))))
        X_test = np.asarray(list(range(len(self.train_df), len(self.train_df) + len(self.test_df))))
        X_test = self.convert_x(X_test)
        for tr_idx, val_idx in folds.split(X):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            X_tr = self.convert_x(X_tr)
            X_val = self.convert_x(X_val)

            mod = sm.OLS(y_tr, X_tr)
            res = mod.fit()
            oof_preds[val_idx] = qt.inverse_transform(res.predict(X_val).reshape((-1, 1))).ravel()

            test_preds.append(qt.inverse_transform(res.predict(X_test).reshape((-1, 1))).ravel())

        test_preds = np.mean(test_preds, axis = 0)
        
        return test_preds, oof_preds, res