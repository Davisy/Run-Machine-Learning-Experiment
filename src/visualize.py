import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools


# function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues, fig_num=None):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""

	if fig_num is not None:
		plt.subplot(2, 2, fig_num)
	fmt = 'd'
	cm = confusion_matrix(y_true, y_pred)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=90)
	plt.yticks(tick_marks, classes)
	plt.title(title)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


# function save the confusion matrix plot in pdf
def savefig(filename, leg=None, format='.pdf', *args, **kwargs):

	if leg:
		art = [leg]
		plt.savefig(filename + format, additional_artists=art, bbox_inches="tight", *args, **kwargs)
	else:
		plt.savefig(filename + format, bbox_inches="tight", *args, **kwargs)
	plt.close()