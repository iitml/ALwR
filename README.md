# ALwR
Active Learning with Rationales


This is the source code for the paper "Active Learning with Rationales for Text Classification" (Manali Sharma, Di Zhuang, Mustafa Bilgic), In North American Chapter of the Association for Computational Linguistics – Human Language Technologies, 2015

'active_learning_with_rationales.py' implements the code to start all the experiments. The file accepts the following command-line arguments to run various experiments:


1. -dataset: Dataset to be used for an experiment. The four datasets used in the paper are IMDB ('imdb') NOVA ('nova'), SRAA ('SRAA'), WvsH (20newsgroups; it must have 2 valid group names. 
	The groups use in the paper are comp.os.ms-windows.misc comp.sys.ibm.pc.hardware)

2. -tfidf: If specified as true, performs tf-idf transformation of the dataset

3. -metric: The feature expert ranks features based on the specified metric. Currently supported options are:	
	(i) Chi Squared statistic (chi2). This statistic is used in the paper.
	(i) Mutual Information (mutual_info)
	(iii) Ranking based on feature weights obtained by training an logistic regression with L1 regularization (L1)
                        
4. -c: Penalty term for the L1 feature expert

5. -debug: If debug is ture, it enables debugging of the code

6. -trials: Number of trials to run for each experiment

7. -seed: Seed to the random number generator

8. -bootstrap: Number of documents to select randomly for bootstrapping the initial model

9. -balance: Ensures both classes starts with equal # of docs after bootstrapping

10. -budget: Budget (in terms of number of documents) for each experiment

11. -step_size: Number of documents to label at each iteration of active learning

12. -strategy: Active learning strategy to use for selecting documents. Currently supported active learning strategies include random sampling (RND), uncertainty sampling (UNC), 
	uncertainty sampling strategy that prefers documents with conflicting rationales (UNC_PC) and uncertainty sampling strategy that prefers documents with no conflicting rationales (UNC_PNC)

13. -topk_unc: Number of uncertain documents to consider to differentiate between types of uncertainties

14. -w_o: The 'o' parameter in the paper. This is the weight of all features other than rationales

15. -w_r: The 'r' parameter in the paper. This is the weight of all rationale features for a document

16. -model_type: Type of classifier to be used. Currently supported options include logistic regression with L2 regularization (lrl2), logistic regression with L1 regularization (lrl1), 
	Multinomial naive Bayes (mnb), support vector machines (svm_linear), Strategy presented in Melville et al 2009 paper (Melville_etal), Strategy presented in Zaidan et al 2007 paper (Zaidan_etal)

17. -alpha: Smooting parameter for the MultinomialNB instance model

18. -lr_C': Penalty term for the logistic regression classifier

19. -svm_C: Penalty term for the SVM classifier


PARAMETERS FOR THE APPROACH PRESENTED IN Melville etal PAPER

20. -Meville_etal_r: The 'r' parameter for the feature model in Melville  et al 2009 paper

21. -instance_model_weight: Weight for the instance model in Melville  et al 2009 paper. Note that weight for the feature model will be 1. - instance_model_weight')

PARAMETERS FOR THE APPROACH PRESENTED IN Zaidan etal PAPER        

23. -Zaidan_etal_Ccontrast: Parameter Ccontrast in Zaidan et al 2007 paper

24. -Zaidan_etal_C: Parameter C in Zaidan et al 2007 paper

25. -Zaidan_etal_mu: Parameter Ccontrast in Zaidan et al 2007 paper


26. -file_tagThe additional tag you might want to give to the saved file

 
