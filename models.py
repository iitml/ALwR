import sys
import os
sys.path.append(os.path.abspath("."))

import numpy as np
from sklearn.naive_bayes import MultinomialNB

np.seterr(divide='ignore')

class FeatureMNBUniform(MultinomialNB):
    def __init__(self, class0_features, class1_features, num_feat, class_prior = [0.5, 0.5], r=100.):
        self.class0_features = list(class0_features)
        self.class1_features = list(class1_features)        
        self.num_features = num_feat
        self.class_prior = class_prior
        self.r = r
    
    def update(self):
        unlabeled_features = set(range(self.num_features))
        unlabeled_features.difference_update(self.class0_features)
        unlabeled_features.difference_update(self.class1_features)
        unlabeled_features = list(unlabeled_features)

        n0 = len(self.class0_features) # p
        n1 = len(self.class1_features) # n
        nu = len(unlabeled_features) # m-p-n        

        self.feature_log_prob_ = np.zeros(shape=(2,self.num_features))

        if self.class0_features != []:
            self.feature_log_prob_[0][self.class0_features] = np.log(1./(n0+n1)) # Equation 12
            self.feature_log_prob_[1][self.class0_features] = np.log(1./((n0+n1)*self.r)) # Equation 13
        
        if self.class1_features != []:
            self.feature_log_prob_[1][self.class1_features] = np.log(1./(n0+n1)) # Equation 12
            self.feature_log_prob_[0][self.class1_features] = np.log(1./((n0+n1)*self.r)) # Equation 13
        
        #Equation 14
        self.feature_log_prob_[0][unlabeled_features] = np.log((n1*(1-1./self.r))/((n0+n1)*nu))
        self.feature_log_prob_[1][unlabeled_features] = np.log((n0*(1-1./self.r))/((n0+n1)*nu))

        self.class_log_prior_ = np.log(self.class_prior)
        self.classes_ = np.array([0, 1])

    def fit(self, feature, label):
        if label == 0:
            new_class0_features = set(self.class0_features)
            new_class0_features.update([feature])
            self.class0_features = list(new_class0_features)
        else:
            new_class1_features = set(self.class1_features)
            new_class1_features.update([feature])
            self.class1_features = list(new_class1_features)
        
        self.update()



class PoolingMNB(MultinomialNB):
    def fit(self, mnb1, mnb2, weights=[0.5, 0.5]):
        self.feature_log_prob_ = np.log(weights[0]*np.exp(mnb1.feature_log_prob_) + \
                                        weights[1]*np.exp(mnb2.feature_log_prob_))
        self.class_log_prior_ = mnb1.class_log_prior_
        self.classes_ = mnb1.classes_
