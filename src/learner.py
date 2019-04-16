import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from data import load_train_data, load_valid_data, load_test_data, load_test_labels
from logs import log
from visualize import plot_confusion_matrix, savefig
import os
import yaml
import matplotlib.pyplot as plt

#set logger
logger = log("mylogs.log")

# folder contain configuration file
CONFIG = "../config/"

# folder contain models
MODELS = "../models/"

class_names = ["continue", "drop"]

#load yaml configuration files
def load_config(config_name):

	config_name = "{}.yaml".format(config_name)
	with open(os.path.join(CONFIG,config_name)) as file:
		config = yaml.safe_load(file)

	return config

#function to train the selected classifier
def train(classifier, config_file):

	#load train dataset
	train = load_train_data("../data/data_for_train.csv")

	#list of features column
	feature_columns = ['gender', 'caste', 'mathematics_marks',
				   'english_marks', 'science_marks', 'science_teacher',
				   'languages_teacher', 'guardian', 'internet']

	y = train['continue_drop'].values
	X = train[feature_columns].values

	if classifier not in ["LG", "KNN"]:
		print("Sorry Your {} Classifer was not found".format(classifier))
	else:
		# function to load the classifier
		def load_classifier(classifier,model=None):

			if classifier == "KNN":
				# load parameters from configuration file
				parameters = load_config(config_file)

				model = KNeighborsClassifier(n_neighbors= parameters['n_neighbors'], weights=parameters['weights'],
									 algorithm=parameters['algorithm'], leaf_size=parameters['leaf_size'],
									 n_jobs=parameters['n_jobs'])

			else:

				# load parameters from configuration file
				parameters = load_config(config_file)

				model = LogisticRegression(penalty=parameters['penalty'], max_iter=parameters['max_iter'],
											n_jobs=parameters['n_jobs'],C=parameters['C'])
			return model

		model = load_classifier(classifier)

		print("Start Training {} Classifier with {}".format(classifier, config_file))
		logger.info("Start Training {} Classifier with {}".format(classifier, config_file))

		# training model
		model.fit(X, y)

		#save the model
		model_name= "{}_{}".format(classifier,config_file)
		joblib.dump(model, '../models/{}.pkl'.format(model_name))

		print("Model Name is:{}.pkl".format(model_name))
		print("Training {} classifier ends".format(classifier))
		logger.info("Model Name is:{}.pkl".format(model_name))
		logger.info("************ Training {} classifier ends **************".format(classifier))


# test your trained model in validation set
def validate(model_name):

	#load train dataset
	valid = load_valid_data("../data/data_for_validation.csv")

	#list of features column
	feature_columns = ['gender', 'caste', 'mathematics_marks',
				   'english_marks', 'science_marks', 'science_teacher',
				   'languages_teacher', 'guardian', 'internet']

	y = valid['continue_drop'].values
	X = valid[feature_columns].values

	#check if the model exist
	real_model_name = model_name + ".pkl"
	file_path  = os.path.join(MODELS,real_model_name)
	if not os.path.isfile(file_path):
		print("Sorry your {} model was no found".format(real_model_name))
		logger.warning("You tried to run a {} model which was not found in your models directory".format(real_model_name))

	else:
		# load the model from disk
		model = joblib.load(os.path.join(MODELS,real_model_name))

		print("Start Validation by using {} model".format(model_name))
		logger.info("Start Validation by using {} model".format(model_name))

		y_pred = model.predict(X)
		fscore = f1_score(y, y_pred, average='weighted')

		#draw and save confusion matrix
		plot_confusion_matrix(y,y_pred, class_names, title = model_name + "_valid_cm")
		plt.savefig("../figures/{}_{}_cm.pdf".format(model_name, "valid"), bbox_inches="tight")
		plt.close()

		print("F1 Score for {0} model is {1:.3f}".format(model_name, fscore))
		print("Validation Ends")
		logger.info(" mode: {0}, Model: {1}, F1 score: {2:.3f}".format("valid",model_name,fscore))
		logger.info("************* Validation Ends *********************")


## testing function
def predict(model_name):

	# load test data
	test = load_test_data("../data/test_data.csv")
	test = test.values

	#load test labels
	labels = load_test_labels("../data/test_label.csv")
	labels = labels['continue_drop'].values

	# check if the model exist
	real_model_name = model_name + ".pkl"
	file_path  = os.path.join(MODELS,real_model_name)
	if not os.path.isfile(file_path):
		print("Sorry your {} model was no found".format(real_model_name))
		logger.warning("You tried to run a {} model which was not found in your models directory".format(real_model_name))

	else:
		# load the model from disk
		model = joblib.load(os.path.join(MODELS,real_model_name))

		print("Start Testing by using {} model".format(model_name))
		logger.info("Start Testing  by using {} model".format(model_name))

		y_pred = model.predict(test)
		fscore = f1_score(labels, y_pred, average="weighted")

		#draw and save confusion matrix
		plot_confusion_matrix(labels,y_pred, class_names,title = model_name + "_test_cm")
		plt.savefig("../figures/{}_{}_cm.pdf".format(model_name, "test"), bbox_inches="tight")
		plt.close()

		print("F1 Score for {0} model is {1:.3f}".format(model_name, fscore))
		print("Testing Ends")
		logger.info(" mode: {0}, Model: {1}, F1 score: {2:.3f}".format("Test",model_name,fscore))
		logger.info("************* Testing Ends *********************")


