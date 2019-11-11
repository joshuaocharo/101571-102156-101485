
#ID3 Algorithm Illustration for Decision Trees
import os
import math
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import tree, metrics
import networkx as nx
import matplotlib.pyplot as plt
import operator
from sklearn.model_selection import train_test_split


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



from IPython.display import Image
import pydotplus
import math
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import tree, metrics
import networkx as nx
import matplotlib.pyplot as plt
import operator
from sklearn.model_selection import train_test_split




data=pd.read_csv('data1.csv',names=['engine','turbo','weight','fueleco','fast'])
#data.head()
data.info() #view details about imported data

data['fast'],class_names = pd.factorize(data['fast'])
print(class_names)
print(data['fast'].unique())


#factorize data to interger format
data['engine'],_ = pd.factorize(data['engine'])
data['turbo'],_ = pd.factorize(data['turbo'])
data['weight'],_ = pd.factorize(data['weight'])
data['fueleco'],_ = pd.factorize(data['fueleco'])

#print the intergers optional
print(data)
#print(data['engine'].unique())
#print(data['turbo'].unique())
#print(data['weight'].unique())
#print(data['fueleco'].unique())
#data.head()

data.info() #check the data info should be converted to intergers

X = data.iloc[:,:-1]
y = data.iloc[:,-1]


# split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(X_train, y_train)

# use the model to make predictions with the test data
y_pred = dtree.predict(X_test)

# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

print(y_pred)

#confusion matrix



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
'''plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')'''

plt.show()