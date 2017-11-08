'''# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
# Load libraries
'''
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names = names)

#shape of data set (instances and attributes)
#print(dataset.shape)

#print the first 20 rows of data
#print (dataset.head(20))

#describes the data set in statistical means
#print(dataset.describe())

#print(dataset.groupby('class').size())

#print (dataset.head(150))
#box and whisker plots // BETTER UNDERSTAND INDIVIDUAL ATTRUBUTES
#dataset.plot(kind = 'box',subplots = True, layout= (2,2),sharex = False, sharey = False)
#BETTER UNDERSTAND INDIVIDUAL ATTRUBUTES
#histogram plot
#dataset.hist()
#scatter plot matrix
# BETTER UNDERSTAND RELATIONSHIPS BETWEEN ATTRIBUTES
#scatter_matrix(dataset)


array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size = validation_size, random_state = seed)

seed = 7
scoring = 'accuracy'

models = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append()
