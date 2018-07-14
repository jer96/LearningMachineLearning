from sklearn import tree
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import math

clf = tree.DecisionTreeClassifier()
# CHALLENGE - create 3 more classifiers...
# 1
svm_clf = svm.SVC()
# 2
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
# 3
nn_clf = clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=.2)

# CHALLENGE - ...and train them on our data
clf = clf.fit(X_train, y_train)
svm_clf = svm_clf.fit(X_train, y_train)
gb_clf = gb_clf.fit(X_train, y_train)
nn_clf = nn_clf.fit(X_train, y_train)

classifiers = ['decision tree', 'svm', 'gradient boost', 'neural network']
clfs = [clf, svm_clf, gb_clf, nn_clf]

# CHALLENGE compare their reusults and print the best one!
accuracy_scores = []

for clf in clfs: 
    accuracy_scores.append(clf.score(X_test, y_test))

max_accr = max(accuracy_scores)
max_clf = classifiers[accuracy_scores.index(max_accr)]
max_accr = round((max_accr * 100), 2)

print('the {} classifier was the best with accuracy of {}%'.format(max_clf, max_accr))

# to_pred = [[190, 70, 43], [190, 70, 30]]
# prediction = clf.predict(to_pred)
# svm_pred = svm_clf.predict(to_pred)
# gb_pred = gb_clf.predict(to_pred)
# nn_pred = nn_clf.predict(to_pred)
# print(prediction)
# print(svm_pred)
# print(gb_pred)
# print(nn_pred)
