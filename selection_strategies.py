'''
selection_strategies.py

This file contains the following:
    RandomBootstrap: pick instances randomly for bootstrapping the initial model
    RandomStrategy: pick the next document randomly from pool
    UNCSampling: pick the next document based on uncertainty
    UNCPreferNoConflict: pick the next document out of top k uncertain instances, which does not contain conflicting rationales
    UNCPreferConflict: pick the next document out of top k uncertain instances, which contains conflicting rationales    
'''

import sys
import os
sys.path.append(os.path.abspath("."))

#import sys
from time import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from imdb import load_imdb
from feature_expert import print_all_features
from sklearn.naive_bayes import MultinomialNB
#from models import FeatureMNBUniform, FeatureMNBWeighted, PoolingMNB
from sklearn import metrics

def RandomBootstrap(X_pool, y_pool, size, balance, seed=0):
    '''
    Assume the task is binary classification
    '''
    print '-' * 50
    print 'Starting bootstrap...'
    print 'Initial training set size = %d' % size
    start = time()    
    
    random_state = np.random.RandomState(seed=seed)
    poolsize = y_pool.shape[0]
    
    pool_set = np.arange(poolsize)
    
    if balance: # select 1/2 * size from each class
        class0_size = size / 2
        class1_size = size - class0_size
        class0_indices = np.nonzero(y_pool == 0)[0]
        class1_indices = np.nonzero(y_pool == 1)[0]
        
        class0_docs = random_state.permutation(class0_indices)[:class0_size]
        class1_docs = random_state.permutation(class1_indices)[:class1_size]
        
        training_set = np.hstack((class0_docs, class1_docs))
        
    else: # otherwise, pick 'size' documents randomly
        training_set = random_state.permutation(pool_set)[:size]
    
    pool_set = np.setdiff1d(pool_set, training_set)
    
    print 'bootstraping took %0.2fs.' % (time() - start)
    
    return (training_set.tolist(), pool_set.tolist())

class RandomStrategy(object):
    def __init__(self, seed=0):
        self.rgen = np.random.RandomState(seed)

    def choices(self, model, X, pool, k): 
        return self.rgen.permutation(pool)[:k]
        

class UNCSampling(object):
    '''
    This class performs uncertainty sampling based on the model.
    '''
    
    def choices(self, model, X, pool, k):
        if isinstance(model, MultinomialNB):
            y_probas = model.predict_proba(X[pool])
            uncerts = np.argsort(np.max(y_probas, axis=1))[:k]
        else:
            y_decision = model.decision_function(X[pool])
            uncerts = np.argsort(np.absolute(y_decision))[:k]
        
        return [pool[u] for u in uncerts]


class UNCPreferNoConflict(object):
    '''
    This class picks uncertain instances.
    Whenever it can, it prefers instances that do not have conflicts.
    No conflict means this instance do not have annotated features
    from opposing classes.
    '''

    
    def choices(self, model, X, pool, k, discovered_class0_feats, discovered_class1_feats, top_k):
        
        if isinstance(model, MultinomialNB):
            y_probas = model.predict_proba(X[pool])
            uncerts = np.array(pool)[np.argsort(np.max(y_probas, axis=1))][:top_k]
        else:
            y_decision = model.decision_function(X[pool])
            uncerts = np.array(pool)[np.argsort(np.absolute(y_decision))][:top_k]
        
        
        conflicted = []
        
        chosen = set()
        
        for unc in uncerts[:top_k]:
            x_feats = X[unc].indices
            x_class0_feats = discovered_class0_feats.intersection(x_feats)
            x_class1_feats = discovered_class1_feats.intersection(x_feats)
            
            conflicted.append(len(x_class0_feats) > 0) and (len(x_class1_feats) > 0)
        
        index = 0
        while len(chosen) < k and index < top_k:
            if not conflicted[index]:
                chosen.add(uncerts[index])
            index += 1        
        
        
        if len(chosen) < k: # Not enough un-conflicted could be found
            index = 0        
            while len(chosen) < k and index < top_k:            
                chosen.add(uncerts[index])
                index += 1         
        
        return list(chosen) 
        
class UNCPreferConflict(object):
    '''
    This class picks uncertain instances.
    Whenever it can, it prefers instances that have a conflict.
    A conflict means this instance has annotated features
    from opposing classes.
    '''

    
    def choices(self, model, X, pool, k, discovered_class0_feats, discovered_class1_feats, top_k):
        
        if isinstance(model, MultinomialNB):
            y_probas = model.predict_proba(X[pool])
            uncerts = np.array(pool)[np.argsort(np.max(y_probas, axis=1))][:top_k]
        else:
            y_decision = model.decision_function(X[pool])
            uncerts = np.array(pool)[np.argsort(np.absolute(y_decision))][:top_k]
        
        conflicted = []
        
        chosen = set()
        
        for unc in uncerts[:top_k]:
            x_feats = X[unc].indices
            x_class0_feats = discovered_class0_feats.intersection(x_feats)
            x_class1_feats = discovered_class1_feats.intersection(x_feats)
            
            conflicted.append(len(x_class0_feats) > 0) and (len(x_class1_feats) > 0)
        
        index = 0
        while len(chosen) < k and index < top_k:
            if conflicted[index]:
                chosen.add(uncerts[index])
            index += 1                
        
        if len(chosen) < k: # Not enough conflicted could be found
            index = 0        
            while len(chosen) < k and index < top_k:            
                chosen.add(uncerts[index])
                index += 1         
        
        return list(chosen)
