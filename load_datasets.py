import sys
import os
sys.path.append(os.path.abspath("."))

from time import time
import glob
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split, ShuffleSplit

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_imdb(path, shuffle=True, random_state=42, \
              vectorizer = CountVectorizer(min_df=2, max_df=1.0, binary=False)):
    
    print "Loading the imdb reviews data"
    
    train_neg_files = glob.glob(path+"/train/neg/*.txt")
    train_pos_files = glob.glob(path+"/train/pos/*.txt")
    
    train_corpus = []
    y_train = []
    
    for tnf in train_neg_files:
        f = open(tnf, 'r')
        line = f.read()
        train_corpus.append(line)
        y_train.append(0)
        f.close()
    
    for tpf in train_pos_files:
        f = open(tpf, 'r')
        line = f.read()
        train_corpus.append(line)
        y_train.append(1)
        f.close()
    
    test_neg_files = glob.glob(path+"/test/neg/*.txt")
    test_pos_files = glob.glob(path+"/test/pos/*.txt")
    
    test_corpus = []
    
    y_test = []
    
    for tnf in test_neg_files:
        f = open(tnf, 'r')
        test_corpus.append(f.read())
        y_test.append(0)
        f.close()
    
    for tpf in test_pos_files:
        f = open(tpf, 'r')
        test_corpus.append(f.read())
        y_test.append(1)
        f.close()
    
    print "Data loaded."
    
    print "Extracting features from the training dataset using a sparse vectorizer"
    print "Feature extraction technique is %s." % vectorizer
    t0 = time()
    
    X_train = vectorizer.fit_transform(train_corpus)
    
    duration = time() - t0
    print("done in %fs" % (duration))
    print "n_samples: %d, n_features: %d" % X_train.shape
    print
        
    print "Extracting features from the test dataset using the same vectorizer"
    t0 = time()
        
    X_test = vectorizer.transform(test_corpus)
    
    duration = time() - t0
    print("done in %fs" % (duration))
    print "n_samples: %d, n_features: %d" % X_test.shape
    print
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(y_train))       
        
        X_train = X_train.tocsr()
        X_train = X_train[indices]
        y_train = y_train[indices]
        train_corpus_shuffled = [train_corpus[i] for i in indices]
        
        indices = np.random.permutation(len(y_test))
        
        X_test = X_test.tocsr()
        X_test = X_test[indices]
        y_test = y_test[indices]
        test_corpus_shuffled = [test_corpus[i] for i in indices]
    else:
        train_corpus_shuffled = train_corpus
        test_corpus_shuffled = test_corpus
         
    return X_train, y_train, X_test, y_test, train_corpus_shuffled, test_corpus_shuffled

def load_newsgroups(class1, class2, shuffle=False, random_state=42, remove=('headers', 'footers'), \
                    vectorizer=CountVectorizer(min_df=5, max_df=1.0, binary=False)):
    
    sep = '-' * 50
    print sep
    print "Loading the 20 newsgroup data..."
    cats = [class1, class2]
    newsgroups_train = fetch_20newsgroups(subset='train', remove=remove, categories=cats)
    newsgroups_test = fetch_20newsgroups(subset='test', remove=remove, categories=cats)
    print "Data loaded."
    
    train_corpus = np.array(newsgroups_train.data)
    test_corpus = np.array(newsgroups_test.data)
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target
    
    print sep
    print "Extracting features from the training dataset using a sparse vectorizer"
    print "Feature extraction technique is %s." % vectorizer
    
    t0 = time()
    X_train = vectorizer.fit_transform(train_corpus)
    X_train = X_train.tocsr()
    
    # ensures that no document has 0 features
    non_empty_docs = []
    for i in range(X_train.shape[0]):
        if len(X_train[i].indices) > 0:
            non_empty_docs.append(i)
    
    X_train = X_train[non_empty_docs]
    y_train = y_train[non_empty_docs]
    train_corpus = train_corpus[non_empty_docs]
    
    duration = time() - t0
    print "done in %fs" % duration
    print "n_samples: %d, n_features: %d" % X_train.shape
    
    print sep        
    print "Extracting features from the test dataset using the same vectorizer"
    t0 = time()        
    X_test = vectorizer.transform(test_corpus)
    X_test = X_test.tocsr()
    
    non_empty_docs = []
    for i in range(X_test.shape[0]):
        if len(X_test[i].indices) > 0:
            non_empty_docs.append(i)
    
    X_test = X_test[non_empty_docs]
    y_test = y_test[non_empty_docs]
    test_corpus = test_corpus[non_empty_docs]
    
    duration = time() - t0
    print "done in %fs" % duration
    print "n_samples: %d, n_features: %d" % X_test.shape

    print sep
    print 'train corpus has %d documents' % len(train_corpus)
    print 'test corpus has %d documents' % len(test_corpus)
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(y_train))       
        
        X_train = X_train.tocsr()
        X_train = X_train[indices]
        y_train = y_train[indices]
        train_corpus_shuffled = [train_corpus[i] for i in indices]
        
        indices = np.random.permutation(len(y_test))
        
        X_test = X_test.tocsr()
        X_test = X_test[indices]
        y_test = y_test[indices]
        test_corpus_shuffled = [test_corpus[i] for i in indices]
    else:
        train_corpus_shuffled = train_corpus
        test_corpus_shuffled = test_corpus
    
    return X_train, y_train, X_test, y_test, train_corpus_shuffled, test_corpus_shuffled

def load_nova(filepath='./text_datasets/nova/nova.dat', n_features=16969, test_split=1./3, shuffle=True, rnd=3439):
     
    print '-' * 50
    print "Loading the NOVA dataset..."
    t0 = time()

    X_pool, y_pool = load_svmlight_file(filepath, n_features)

    duration = time() - t0
    print "Loading took %0.2fs." % duration
    
    y_pool[y_pool==-1] = 0
    indices = ShuffleSplit(X_pool.shape[0], n_iter=1, test_size=test_split, indices=True, random_state=rnd)
    for train_ind, test_ind in indices:
        X_train = X_pool[train_ind]
        y_train = y_pool[train_ind]
        X_test = X_pool[test_ind]
        y_test = y_pool[test_ind]
    
    return (X_train, y_train, X_test, y_test)

