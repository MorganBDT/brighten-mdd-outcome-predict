from sklearn import tree
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class MLModel():
    
    @staticmethod
    def model_name():
        pass
    
    def get_model(self):
        return self.model
    
    def fit(self, predictors, labels):
        self.model.fit(predictors, labels)
        
    def predict(self, predictors):
        return self.model.predict(predictors)[0]
    
    def proba(self, predictors):
        return self.model.predict_proba(predictors)
    
    def acc(self, predictors, true_labels):
        return self.model.score(predictors, true_labels)
    
    def auc(self, predictors, true_labels):
        return roc_auc_score(self.proba(predictors), true_labels)
    
    def score(self, X, y):
        return self.auc(X, y)
    
    def get_feature_importances(self, predictors):
        pass
    
    @staticmethod
    def plot_feature_importances(feature_imps, predictors, title=None):
        sns.barplot(x=feature_imps, y=feature_imps.index)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        if title is None:
            plt.title("Visualization of feature importance")
        else:
            plt.title(title)
        plt.show()
        
        
class LogisticRegression(MLModel):
    def __init__(self, model_args=None):
        self.model = SklearnLogisticRegression(solver='liblinear') #, penalty='l1', C=0.05)
    
    @staticmethod
    def model_name():
        return "logistic_regression"
    
    def get_feature_importances(self, predictors):
        return pd.Series(np.abs(np.reshape(self.model.coef_, -1)),index=predictors).sort_values(ascending=False)

        
class DecisionTree(MLModel):
    def __init__(self, model_args):
        self.model = tree.DecisionTreeClassifier(
            max_depth=model_args["dt_max_depth"],
            random_state=model_args.get("random_state")
        )
    
    @staticmethod
    def model_name():
        return "decision_tree"
    
    def get_feature_importances(self, predictors):
        return pd.Series(self.model.feature_importances_,index=predictors).sort_values(ascending=False)
    

class RandomForest(MLModel):
    def __init__(self, model_args):
        self.model = RandomForestClassifier(
            n_estimators=model_args["rf_n_estimators"], 
            max_depth=model_args["rf_max_depth"],
            random_state=model_args.get("random_state")
        )
    
    @staticmethod
    def model_name():
        return "random_forest"
    
    def get_feature_importances(self, predictors):
        return pd.Series(self.model.feature_importances_,index=predictors).sort_values(ascending=False)
    

class SVM(MLModel):
    def __init__(self, model_args=None):
        self.model = svm.SVC(kernel='linear', probability=True)
    
    @staticmethod
    def model_name():
        return "svm"
    
    def get_feature_importances(self, predictors):
        return pd.Series([n**2 for n in self.model.coef_[0]],index=predictors).sort_values(ascending=False)
    

class KNN(MLModel):
    def __init__(self, model_args=None):
        self.model = KNeighborsClassifier(n_neighbors=3)
    
    @staticmethod
    def model_name():
        return "knn"
