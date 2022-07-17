*********This project is on Intent classification of 3 intents dataset only and Sequence labeling************

*********Sequence labeling to find custom entity "Money" from sentences************

Intent Classification:
	models:
	RandomForest , Logistic Regression, DistilBertForTokenClassification
	
	Word Embeddings:
	CountVectorizer, Word2Vec, Senetence BERT


#### Command line arguments:
	--model : the model to use
	--file : select file to do intent classification
	-- embed: select word embedding techniques
	-- text: input text to label sequences

	Ex: python .py --model [model name] --embed [word embeddding] --text [text input] --file [file input]
	
	N.B. use --text only for sequence labeling task


### Command for models:
	Command to select RandomForest model:              rf

	Command to select Logistic Regression:             lr

	Command to select Countvectorizer word embedding:  cv

	Command to select Word2Vec word embedding:         sg

	Command to select sentence bert:                   sbert


### Format to run Intent Classification
	
	# Command to select select randomforst and word2vec model on test data
		python ./intent.py --model rf --embed sg --file three_intents_test_data.csv
	# Command to select select randomforst and sbert model on test data
		python ./intent.py --model rf --embed sbert --file three_intents_test_data.csv
	# Command to select select logistic regression and sbert model on test data
		python ./intent.py --model rf --embed sbert --file three_intents_test_data.csv


### Format to run sequence labeling
	# Command to run sequence labeling on a text input
		1.python ./intent.py --model s-labeling  --text "1000 dollars is a lot of money"
		2.python ./intent.py --model s-labeling  --text "send 100 dollars to my account"
		

N.B. the model for 150 intents is 2.5 GB in total so shared only testing code for 3 intents classification
