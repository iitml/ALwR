import sys
import os
sys.path.append(os.path.abspath("."))

from time import time
import numpy as np
import argparse
import pickle
import scipy.sparse as sp
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn import metrics
from models import FeatureMNBUniform, PoolingMNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from load_datasets import load_imdb, load_newsgroups, load_nova
from feature_expert import feature_expert
from selection_strategies import RandomBootstrap, RandomStrategy, UNCSampling
from selection_strategies import UNCPreferNoConflict, UNCPreferConflict


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.seterr(divide='ignore')
  

def learn(model_type, X_pool, y_pool, X_test, y_test, training_set, pool_set, feature_expert, \
          selection_strategy, budget, step_size, topk, w_o, w_r, seed=0, alpha=1, poolingMNBWeights=[0.5, 0.5], Meville_etal_r=100.0, lr_C=1, svm_C=1, \
          Zaidan_etal_C=1, Zaidan_etal_Ccontrast=1, Zaidan_etal_mu=1, Debug=False):
    
    start = time()
    print '-' * 50
    print 'Starting Active Learning...'
    
    _, num_feat = X_pool.shape
    model_scores = {'auc':[], 'accu':[]}
    
    rationales  = set()
    rationales_c0  = set()
    rationales_c1  = set()

    feature_expert.rg.seed(seed)
    
    num_training_samples = []
    
    number_of_docs = 0
    
    docs = training_set
    
    X_train = None
    y_train = []
    
    if model_type=='poolingMNB':      
        # create feature model  
        classpriors=np.zeros(2)            
        classpriors[1] = (np.sum(y_pool[docs])*1.)/(len(docs)*1.)
        classpriors[0] = 1. - classpriors[1] 

        feature_model = FeatureMNBUniform(rationales_c0, rationales_c1, num_feat, classpriors, Meville_etal_r)    

    for doc_id in docs:
        
        number_of_docs=number_of_docs+1    
        
        feature = feature_expert.any_informative_feature(X_pool[doc_id], y_pool[doc_id])

        if model_type == 'Melville_etal':        
            if feature:
                feature_model.fit(feature, y_pool[doc_id])
        
        rationales.add(feature)

        if y_pool[doc_id] == 0:
            rationales_c0.add(feature)
        else:
            rationales_c1.add(feature)        
                    

        if model_type == 'Zaidan_etal':
            x = sp.csr_matrix(X_pool[doc_id], dtype=np.float64)
            if feature is not None:
                x_pseudo = (X_pool[doc_id]).todense()
                                
                # create pseudoinstances based on rationales provided; one pseudoinstance is created per rationale.
                x_feats = x[0].indices
        
                for f in x_feats:
                    if f == feature:
                        test= x[0,f]
                        x_pseudo[0,f] = x[0,f]/Zaidan_etal_mu
                    else:                                              
                        x_pseudo[0,f] = 0.0                          
                x_pseudo=sp.csr_matrix(x_pseudo, dtype=np.float64)

        else:
            x = sp.csr_matrix(X_pool[doc_id], dtype=float)
            if "poolingMNB" not in model_type:         
                x_feats = x[0].indices
                for f in x_feats:
                    if f == feature:
                        x[0,f] = w_r*x[0,f]
                    else:
                        x[0,f] = w_o*x[0,f]
        

        if model_type=='Zaidan_etal':
            if not y_train:
                X_train = x      
                if feature is not None:      
                    X_train = sp.vstack((X_train, x_pseudo))
            else:
                X_train = sp.vstack((X_train, x))
                if feature is not None:
                    X_train = sp.vstack((X_train, x_pseudo))
        
            y_train.append(y_pool[doc_id])
            if feature is not None:
                # append y label again for the pseudoinstance created
                y_train.append(y_pool[doc_id])
        

            sample_weight.append(Zaidan_etal_C)
            if feature is not None:
                # append instance weight=Zaidan_etal_Ccontrast for the pseudoinstance created
                sample_weight.append(Zaidan_etal_Ccontrast)  

        else:
            if not y_train:
                X_train = x
            else:
                X_train = sp.vstack((X_train, x))
        
            y_train.append(y_pool[doc_id])
    
    # Train the model
    
    if model_type=='lrl2':
        random_state = np.random.RandomState(seed=seed)
        model = LogisticRegression(C=lr_C, penalty='l2', random_state=random_state)
    elif model_type=='lrl1':
        random_state = np.random.RandomState(seed=seed)
        model = LogisticRegression(C=lr_C, penalty='l1', random_state=random_state)        
    elif model_type=='mnb':        
        model = MultinomialNB(alpha=alpha)        
    elif model_type=='svm_linear':
        random_state = np.random.RandomState(seed=seed)
        model = LinearSVC(C=svm_C, random_state=random_state)
    elif model_type=='Melville_etal':
        instance_model=MultinomialNB(alpha=alpha)        
        model = PoolingMNB()
    elif model_type=='Zaidan_etal':
        random_state = np.random.RandomState(seed=seed)        
        model = svm.SVC(kernel='linear', C=svm_C, random_state=random_state)
        
    if model_type=='Melville_etal':                
        #feature_model.fit(feature, y_pool[doc_id])
        instance_model.fit(X_train, y_train)
        model.fit(instance_model, feature_model, weights=poolingMNBWeights) # train pooling_model
    elif model_type=='Zaidan_etal':
        model.fit(X_train, np.array(y_train), sample_weight=sample_weight)
    else:
        model.fit(X_train, np.array(y_train))
    
    
            
    (accu, auc) = evaluate_model(model, X_test, y_test)
    model_scores['auc'].append(auc)
    model_scores['accu'].append(accu)
    
    num_training_samples.append(number_of_docs)
    
    feature_expert.rg.seed(seed)        
    
    if selection_strategy == 'RND':
        doc_pick_model = RandomStrategy(seed)
    elif selection_strategy == 'UNC':
        doc_pick_model = UNCSampling()         
    elif selection_strategy == 'UNC_PNC':
        doc_pick_model = UNCPreferNoConflict()   
    elif selection_strategy == 'UNC_PC':
        doc_pick_model = UNCPreferConflict()    
    else:
        raise ValueError('Selection strategy: \'%s\' invalid!' % selection_strategy)
 
  
    k = step_size  

    while X_train.shape[0] < budget:                

        # Choose a document based on the strategy chosen
        if selection_strategy == 'UNC_PNC':
            doc_ids = doc_pick_model.choices(model, X_pool, pool_set, k, rationales_c0, rationales_c1, topk)       
        elif selection_strategy == 'UNC_PC':
            doc_ids = doc_pick_model.choices(model, X_pool, pool_set, k, rationales_c0, rationales_c1, topk)        
        else:
            doc_ids = doc_pick_model.choices(model, X_pool, pool_set, k)
        
        if doc_ids is None or len(doc_ids) == 0:
            break        
        
        for doc_id in doc_ids:
            # Remove the chosen document from pool and add it to the training set
            pool_set.remove(doc_id)
            training_set.append(doc_id)

            #feature = feature_expert.most_informative_feature(X_pool[doc_id], y_pool[doc_id])
            feature = feature_expert.any_informative_feature(X_pool[doc_id], y_pool[doc_id])

            if model_type=='Melville_etal':        
                if feature:
                    feature_model.fit(feature, y_pool[doc_id])
            
            number_of_docs=number_of_docs+1    

            rationales.add(feature)

            if y_pool[doc_id] == 0:
                rationales_c0.add(feature)
            else:
                rationales_c1.add(feature)
            

            if model_type=='Zaidan_etal':
                x = sp.csr_matrix(X_pool[doc_id], dtype=np.float64)
                if feature is not None:
                    x_pseudo = (X_pool[doc_id]).todense()
                                
                    # create pseudoinstances based on rationales provided; one pseudoinstance is created for each rationale.
                    x_feats = x[0].indices
        
                    for f in x_feats:
                        if f == feature:
                            test= x[0,f]
                            x_pseudo[0,f] = x[0,f]/Zaidan_etal_mu
                        else:                                              
                            x_pseudo[0,f] = 0.0                          
                    x_pseudo=sp.csr_matrix(x_pseudo, dtype=np.float64)

            else:
                x = sp.csr_matrix(X_pool[doc_id], dtype=float)
                if "Melville_etal" not in model_type:         
                    x_feats = x[0].indices
                    for f in x_feats:
                        if f == feature:
                            x[0,f] = w_r*x[0,f]
                        else:
                            x[0,f] = w_o*x[0,f]                                   

            if model_type=='Zaidan_etal':
                X_train = sp.vstack((X_train, x))
                if feature is not None:
                    X_train = sp.vstack((X_train, x_pseudo))
        
                y_train.append(y_pool[doc_id])
                if feature is not None:
                    # append y label again for the pseudoinstance created
                    y_train.append(y_pool[doc_id])        

                sample_weight.append(Zaidan_etal_C)
                if feature is not None:
                    # append instance weight=Zaidan_etal_Ccontrast for the pseudoinstance created
                    sample_weight.append(Zaidan_etal_Ccontrast)  

            else:
                X_train = sp.vstack((X_train, x))        
                y_train.append(y_pool[doc_id])
        
        # Train the model

        
        if model_type=='lrl2':
            random_state = np.random.RandomState(seed=seed)
            model = LogisticRegression(C=lr_C, penalty='l2', random_state=random_state)
        elif model_type=='lrl1':
            random_state = np.random.RandomState(seed=seed)
            model = LogisticRegression(C=lr_C, penalty='l1', random_state=random_state)        
        elif model_type=='mnb':        
            model = MultinomialNB(alpha=alpha)        
        elif model_type=='svm_linear':
            random_state = np.random.RandomState(seed=seed)
            model = LinearSVC(C=svm_C, random_state=random_state)
        elif model_type=='Melville_etal':
            instance_model=MultinomialNB(alpha=alpha)        
            model = PoolingMNB()
        elif model_type=='Zaidan_etal':
            random_state = np.random.RandomState(seed=seed)        
            model = svm.SVC(kernel='linear', C=svm_C, random_state=random_state)                                                          

        if model_type=='Melville_etal':                            
            instance_model.fit(X_train, y_train)
            model.fit(instance_model, feature_model, weights=poolingMNBWeights) # train pooling_model
        elif model_type=='Zaidan_etal':
            model.fit(X_train, np.array(y_train), sample_weight=sample_weight)
        else:
            model.fit(X_train, np.array(y_train))
            
        (accu, auc) = evaluate_model(model, X_test, y_test)
        model_scores['auc'].append(auc)
        model_scores['accu'].append(accu)
        
        num_training_samples.append(number_of_docs)
        
  
    print 'Active Learning took %2.2fs' % (time() - start)
    
    return (np.array(num_training_samples), model_scores)

def load_dataset(dataset):
    if dataset == ['imdb']:        
        vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1,1))                
        X_pool, y_pool, X_test, y_test, _, _, = load_imdb(path='./text_datasets/aclImdb/', shuffle=True, vectorizer=vect)
        return (X_pool, y_pool, X_test, y_test, vect.get_feature_names())
    elif isinstance(dataset, list) and len(dataset) == 3 and dataset[0] == '20newsgroups':
        vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1))
        X_pool, y_pool, X_test, y_test, _, _ = \
        load_newsgroups(class1=dataset[1], class2=dataset[2], vectorizer=vect)
        return (X_pool, y_pool, X_test, y_test, vect.get_feature_names())
    elif dataset == ['sraa']:
        X_pool = pickle.load(open('./text_datasets/SRAA/SRAA_X_train.pickle', 'rb'))
        y_pool = pickle.load(open('./text_datasets/SRAA/SRAA_y_train.pickle', 'rb'))
        X_test = pickle.load(open('./text_datasets/SRAA/SRAA_X_test.pickle', 'rb'))
        y_test = pickle.load(open('./text_datasets/SRAA/SRAA_y_test.pickle', 'rb'))
        feat_names = pickle.load(open('./text_datasets/SRAA_feature_names.pickle', 'rb'))
        return (X_pool, y_pool, X_test, y_test, feat_names)
    elif dataset == ['nova']:
        (X_pool, y_pool, X_test, y_test) = load_nova()
        return (X_pool, y_pool, X_test, y_test, None)
       
def run_trials(model_type, num_trials, dataset, tfidf, selection_strategy, metric, C, alpha, poolingMNBWeights, Meville_etal_r,\
                Zaidan_etal_C, Zaidan_etal_Ccontrast, Zaidan_etal_mu, bootstrap_size, balance, budget, step_size, topk, w_o, w_r, seed=0, lr_C=1, svm_C=1, Debug=False):
    
    (X_pool, y_pool, X_test, y_test, feat_names) = load_dataset(dataset)

        
    if not feat_names:
        feat_names = np.arange(X_pool.shape[1])
    
    feat_freq = np.diff(X_pool.tocsc().indptr)   
    
    fe = feature_expert(X_pool, y_pool, metric, C=C, pick_only_top=True)
    
    tfidft = TfidfTransformer()
    
    if tfidf:
        print "Performing tf-idf transformation"
        X_pool = tfidft.fit_transform(X_pool)
        X_test = tfidft.transform(X_test)        
    
    result = np.ndarray(num_trials, dtype=object)
    
    for i in range(num_trials):
        print '-' * 50
        print 'Starting Trial %d of %d...' % (i + 1, num_trials)

        trial_seed = seed + i # initialize the seed for the trial
        
        training_set, pool_set = RandomBootstrap(X_pool, y_pool, bootstrap_size, balance, trial_seed)
                
        result[i] = learn(model_type, X_pool, y_pool, X_test, y_test, training_set, pool_set, fe, \
                          selection_strategy, budget, step_size, topk, w_o, w_r, trial_seed, alpha, poolingMNBWeights, Meville_etal_r, lr_C, svm_C, Zaidan_etal_C, Zaidan_etal_Ccontrast, Zaidan_etal_mu, Debug)
    
    return result, feat_names, feat_freq

def average_results(result):
    avg_M_scores = dict()
    
    num_trials = result.shape[0]
    
    if num_trials == 1:
        num_training_set, M_scores = result[0]
        return np.array([(num_training_set, M_scores)])
           
    min_training_samples = np.inf
    for i in range(num_trials):
        # result[i][0] is the num_training_set
        min_training_samples = min(result[i][0].shape[0], min_training_samples)
    
    for i in range(num_trials):
        num_training_set, M_scores = result[i]
        if i == 0:
            avg_M_scores['accu'] = np.array(M_scores['accu'])[:min_training_samples]
            avg_M_scores['auc'] = np.array(M_scores['auc'])[:min_training_samples]
        else:
            avg_M_scores['accu'] += np.array(M_scores['accu'])[:min_training_samples]
            avg_M_scores['auc'] += np.array(M_scores['auc'])[:min_training_samples]
           
    num_training_set = num_training_set[:min_training_samples]
    avg_M_scores['accu'] = avg_M_scores['accu'] / num_trials
    avg_M_scores['auc'] = avg_M_scores['auc'] / num_trials
    
    return np.array([(num_training_set, avg_M_scores)])

def save_result(result, filename='result.txt'):
    # Saves the data the following order:
    # training sample index, IM_accu, FM_accu, PM_accu, IM_acu, FM_auc, PM_auc, c0_features_discovered so far,
    # c1_features_discovered so far, num_docs_covered, and transition phases for cover_then_disagree approach
    # if the approach is not cover_then_disagree, no transition is saved
    print '-' * 50
    print 'saving result into \'%s\'' % filename
    
    ls_all_results = []
    with open(filename, 'w') as f:
        for i in range(result.shape[0]):
            num_training_set, model_scores = result[i]

            ls_all_results.append(num_training_set)
            ls_all_results.append(model_scores['accu'])
            ls_all_results.append(model_scores['auc'])
            
        header = 'train#\tM_accu\tM_auc'
        f.write('\t'.join([header]*result.shape[0]) + '\n')
        for row in map(None, *ls_all_results):
            f.write('\t'.join([str(item) if item is not None else ' ' for item in row]) + '\n')


def evaluate_model(model, X_test, y_test):        
    if isinstance(model, MultinomialNB):
        y_probas = model.predict_proba(X_test)
        auc = metrics.roc_auc_score(y_test, y_probas[:, 1])
    else: 
        y_decision = model.decision_function(X_test)
        auc = metrics.roc_auc_score(y_test, y_decision)
    if isinstance(model, PoolingMNB):
        pred_y = model.classes_[np.argmax(y_probas, axis=1)]
    else:
        pred_y = model.predict(X_test)
    accu = metrics.accuracy_score(y_test, pred_y)
    return (accu, auc)
    

def save_result_num_a_feat_chosen(result, feat_names, feat_freq):
    
    ave_num_a_feat_chosen = np.zeros(len(feat_names))
    
    for i in range(args.trials):
        num_a_feat_chosen = result[i][-1]
        ave_num_a_feat_chosen += (num_a_feat_chosen / float(args.trials))
    
    filename='_'.join(['_'.join(args.dataset), args.strategy, args.metric, 'w_r={:0.2f}'.format(args.w_r), 'w_o={:0.2f}'.format(args.w_o), "num_a_feat_chosen", 'batch-result.txt'])
    
    print '-' * 50
    print 'saving result into \'%s\'' % filename
            
    with open(filename, 'w') as f:
        f.write("ID\tNAME\tFREQ\tCOUNT\n")
        for i in range(len(feat_names)):
            f.write(str(i)+"\t"+feat_names[i].encode('utf8')+"\t"+str(feat_freq[i])+"\t"+str(ave_num_a_feat_chosen[i])+"\n")

if __name__ == '__main__':
    '''
    Example: 
    To run covering approach with L1 feature expert for 10 trials, no bootstrap and a budget of $100:
    python active_learn.py -strategy covering_fewest -metric L1 -trials 10 -bootstrap 0 -budget 100
    python active_learn.py -dataset 20newsgroups comp.graphics comp.windows.x -strategy random -metric L1 -trials 10 -bootstrap 0 -budget 500
    python active_learn.py -dataset 20newsgroups alt.atheism talk.religion.misc -strategy random -metric mutual_info -trials 10 -bootstrap 0 -budget 500
    python active_learn.py -dataset 20newsgroups alt.atheism talk.religion.misc -strategy cover_then_disagree -metric L1 -trials 10 -bootstrap 0 -budget 500
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default=['imdb'], nargs='*', \
                        help='Dataset to be used: [\'imdb\', \'20newsgroups\'] 20newsgroups must have 2 valid group names')
    parser.add_argument('-strategy', choices=['RND', 'UNC', 'UNC_PNC', 'UNC_PC'], default='RND', help='Document selection strategy to be used')
    parser.add_argument('-metric', choices=['mutual_info', 'chi2', 'L1'], default="L1", \
                        help='Specifying the type of feature expert to be used')
    parser.add_argument('-c', type=float, default=0.1, help='Penalty term for the L1 feature expert')
    parser.add_argument('-debug', action='store_true', help='Enable Debugging')
    parser.add_argument('-trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('-seed', type=int, default=0, help='Seed to the random number generator')
    parser.add_argument('-bootstrap', type=int, default=2, help='Number of documents to bootstrap')
    parser.add_argument('-balance', default=True, action='store_false', help='Ensure both classes starts with equal # of docs after bootstrapping')
    parser.add_argument('-budget', type=int, default=500, help='budget in $')
    parser.add_argument('-alpha', type=float, default=1, help='alpha for the MultinomialNB instance model')
    parser.add_argument('-instance_model_weight', type=float, default=1, help='weight for the instance model in Melville etal paper. Note that weight for feature model will be 1 - instance_model_weight')
    parser.add_argument('-w_o', type=float, default=1., help='The weight of all features other than rationales')
    parser.add_argument('-w_r', type=float, default=1., help='The weight of all rationales for a document')
    parser.add_argument('-step_size', type=int, default=1, help='number of documents to label at each iteration')
    parser.add_argument('-topk_unc', type=int, default=20, help='number of uncertain documents to consider to differentiate between types of uncertainties')
    parser.add_argument('-model_type', choices=['lrl2', 'lrl1', 'mnb', 'svm_linear', 'Melville_etal', 'Zaidan_etal'], default='lrl2', help='Type of classifier to be used')
    parser.add_argument('-lr_C', type=float, default=1, help='Penalty term for the logistic regression classifier')
    parser.add_argument('-svm_C', type=float, default=1, help='Penalty term for the SVM classifier')
    parser.add_argument('-tfidf', default=False, action='store_true', help='Perform tf-idf transformation [default is false]')
    parser.add_argument('-file_tag', default='', help='the additional tag you might want to give to the saved file')    
    parser.add_argument('-Meville_etal_r', type=float, default=100., help='r parameter in Melville etal paper')
    parser.add_argument('-Zaidan_etal_Ccontrast', type=float, default=1, help='Ccontrast in Zaidan etal paper')
    parser.add_argument('-Zaidan_etal_C', type=float, default=1, help='C in Zaidan etal paper')
    parser.add_argument('-Zaidan_etal_mu', type=float, default=1, help='nu in Zaidan etal paper')        

    args = parser.parse_args()
    
    poolingMNBWeights=np.zeros(2)
    
    poolingMNBWeights[0]=args.instance_model_weight
    poolingMNBWeights[1]=1.0-poolingMNBWeights[0]

    result, feat_names, feat_freq = run_trials(model_type=args.model_type, num_trials=args.trials, dataset=args.dataset, tfidf=args.tfidf, selection_strategy=args.strategy,\
                metric=args.metric, C=args.c, alpha=args.alpha, poolingMNBWeights=poolingMNBWeights, Meville_etal_r=args.Meville_etal_r, \
                Zaidan_etal_C=args.Zaidan_etal_C, Zaidan_etal_Ccontrast=args.Zaidan_etal_Ccontrast, Zaidan_etal_mu=args.Zaidan_etal_mu, bootstrap_size=args.bootstrap, balance=args.balance, \
                budget=args.budget, step_size=args.step_size, topk=args.topk_unc, \
                w_o=args.w_o, w_r=args.w_r, seed=args.seed, lr_C=args.lr_C, svm_C=args.svm_C, Debug=args.debug)
    
    print result
    
    for res in result:
        nt, per = res
        accu = per['accu']
        auc = per['auc']
        for i in range(len(nt)):
            print "%d\t%0.4f\t%0.4f" %(nt[i], accu[i], auc[i])
    
    
    if args.model_type=='Melville_etal':
        save_result(average_results(result), filename='_'.join(['Melville_etal', args.dataset[0], 'tfidf{}'.format(args.tfidf), args.file_tag, args.model_type, args.strategy, args.metric, 'alpha={:3.6f}'.format(args.alpha), 'Meville_etal_r={:6.1f}'.format(args.Meville_etal_r), 'IMweight={:1.3f}'.format(poolingMNBWeights[0]), 'FMweight={:1.3f}'.format(poolingMNBWeights[1]), 'averaged', 'batch-result.txt']))
        save_result(result, filename='_'.join(['Melville_etal', args.dataset[0], 'tfidf{}'.format(args.tfidf), args.file_tag, args.model_type, args.strategy, args.metric, 'alpha={:3.6f}'.format(args.alpha),'Meville_etal_r={:6.1f}'.format(args.Meville_etal_r), 'IMweight={:1.3f}'.format(poolingMNBWeights[0]), 'FMweight={:1.3f}'.format(poolingMNBWeights[1]),'all', 'batch-result.txt']))
    elif args.model_type=='Zaidan_etal':
        save_result(average_results(result), filename='_'.join(['Zaidan_etal', args.dataset[0], 'tfidf{}'.format(args.tfidf), args.model_type, args.strategy, args.metric, 'alpha={:2.6f}'.format(args.alpha), 'Zaidan_etal_C={:5.3f}'.format(args.Zaidan_etal_C), 'Zaidan_etal_Ccontrast={:5.3f}'.format(args.Zaidan_etal_Ccontrast), 'Zaidan_etal_mu={:5.3f}'.format(args.Zaidan_etal_mu),  'averaged', 'batch-result.txt']))
        save_result(result, filename='_'.join(['Zaidan_etal', args.dataset[0], 'tfidf{}'.format(args.tfidf), args.model_type, args.strategy, args.metric, 'alpha={:2.6f}'.format(args.alpha), 'Zaidan_etal_C={:5.3f}'.format(args.Zaidan_etal_C), 'Zaidan_etal_Ccontrast={:5.3f}'.format(args.Zaidan_etal_Ccontrast), 'Zaidan_etal_mu={:5.3f}'.format(args.Zaidan_etal_mu), 'all', 'batch-result.txt']))
    else:        
        save_result(average_results(result), filename='_'.join([args.dataset[0], 'tfidf{}'.format(args.tfidf), args.file_tag, args.model_type, args.strategy, args.metric, 'alpha={:2.6f}'.format(args.alpha), 'lr_C={:5.3f}'.format(args.lr_C), 'SVM_C={:5.3f}'.format(args.svm_C), 'w_r={:2.3f}'.format(args.w_r), 'w_o={:2.3f}'.format(args.w_o), 'averaged', 'batch-result.txt']))
        save_result(result, filename='_'.join([args.dataset[0], 'tfidf{}'.format(args.tfidf), args.file_tag, args.model_type, args.strategy, args.metric, 'alpha={:2.6f}'.format(args.alpha), 'lr_C={:5.3f}'.format(args.lr_C), 'SVM_C={:5.3f}'.format(args.svm_C), 'w_r={:2.3f}'.format(args.w_r), 'w_o={:2.3f}'.format(args.w_o), 'all', 'batch-result.txt']))


