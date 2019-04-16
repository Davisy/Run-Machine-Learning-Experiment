import argparse
from learner import train, validate, predict


parser = argparse.ArgumentParser(description='ML for Dropout prediction')
parser.add_argument("--mode", type=str, default="train", help="set module into train or valid or predict mode")
parser.add_argument("--classifier", type=str, default="KNN", help="set classifier either LG or KNN")
parser.add_argument("--model", type=str, default="KNN", help="Select the model to run in your valid mode or "
																 "test model")
parser.add_argument("--config_file", type=str, default="KNN_config",help="set the configuration file")


args = parser.parse_args()


if args.mode == 'train':

	# call train function
	train(args.classifier, args.config_file)

elif args.mode == 'valid':

	# call validate function
	validate(args.model)
elif args.mode == "predict":

	# call predict function
	predict(args.model)

else:
	print("Sorry your  {} mode was not found!!".format(args.mode))



