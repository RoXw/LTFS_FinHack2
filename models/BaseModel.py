class Base_Model(object):
    
    def __init__(self, train_df, test_df, val_df, features, categoricals=[], params_dict = {}, verbose=True):
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.features = features
        self.categoricals = categoricals
        self.target = 'case_count'
        self.verbose = True
        self.params = params_dict
        self.y_pred, self.oof_pred, self.model = self.fit()
                
    def train_model(self, train_set, val_set):
        raise NotImplementedError
            
    def get_params(self):
        raise NotImplementedError
        
    def convert_dataset(self, x_train, y_train):
        raise NotImplementedError
        
    def convert_x(self, x):
        return x
    
    def loss(self, y_true, y_pred):
        raise NotImplementedError
    
    def plot_feature_importance(self):
        raise NotImplementedError
        
    def fit(self):
        raise NotImplementedError