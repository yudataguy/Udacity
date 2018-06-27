#!/usr/bin/python

import sys
import numpy as np
import pandas as pd
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'total_payments', 'loan_advances', 'bonus',
                 'total_stock_value', 'expenses',
                 'from_poi_to_this_person', 'from_this_person_to_poi', 
                 ] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0 ) #remove TOTAL row
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0) #who is this? 
data_dict.pop('LOCKHART EUGENEE', 0) #all empty rows

### Task 3: Create new feature(s)
df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))

# set the index of df to be the employees series:
df.set_index(employees, inplace=True)
df = df.apply(pd.to_numeric, errors = 'coerce')
df.fillna(0, inplace=True)  
df['total_poi_email'] = df['from_poi_to_this_person'] - df['from_this_person_to_poi']
df['total_poi_email_percent'] = df['total_poi_email'] / (df['to_messages'] + df['from_messages'])
df = df.drop('from_poi_to_this_person', 1)
df = df.drop('from_this_person_to_poi', 1)
df = df.drop('to_messages', 1)
df = df.drop('from_messages', 1)
df = df.apply(pd.to_numeric, errors = 'coerce')
df.fillna(0, inplace=True)
new_features_list = list(df.columns.values)

# Ensure POI is the 1st column 
new_features_list.remove('poi')
new_features_list.insert(0, 'poi')

# Remove unnecessary features
old_features = ['deferral_payments', 'deferred_income', 'director_fees', 'email_address',
                         'exercised_stock_options', 'expenses', 'loan_advances', 'long_term_incentive',
                         'other', 'restricted_stock', 'restricted_stock_deferred', 
                         'shared_receipt_with_poi']
for feat in old_features:
    new_features_list.remove(feat)

# create a dictionary from the dataframe
df_dict = df.to_dict('index')

### Store to my_dataset for easy export below.
my_dataset = df_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(features)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, random_state=0)
kmeans.fit(features, labels).transform(features)

#use SelectKBest to pick out the best features
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(k=4).fit(features,labels)
scores = selector.scores_
print(scores)

features_mask = selector.get_support()
new_features = []
for bool, feature in zip(features_mask, new_features_list):
    if bool:
        new_features.append(feature)
print(new_features)

#make train/test sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Pipleline to improve workflow
### selectKbest then classifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(min_samples_split = 10)
select = SelectKBest(k=2)

steps = [('feature_selection', select),
        ('decision_tree', clf)]

pipeline = Pipeline(steps)
pipeline.fit(features, labels)
prediction = pipeline.predict(features_test)
report = classification_report(prediction, labels_test)

parameters = dict(feature_selection__k=[2, 3, 4, 5, 'all'], 
              decision_tree__min_samples_split=[2, 3, 4, 5, 10])

# Validation method - StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(
    labels,
    n_iter = 100,
    test_size = 0.3,
    random_state = 0
    )

cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1', cv = sss)

cv.fit(features, labels)
prediction = cv.predict(features_test)
report = classification_report(prediction, labels_test)
print(report)

print("Best score: %0.3f" % cv.best_score_)
print("Best parameters set:")
best_parameters = cv.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

print("\nThis is the best classifer result.")
from tester import test_classifier
test_classifier(pipeline, my_dataset, new_features_list)

#this is the chosen classifiers
clf = pipeline

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, new_features_list)

print("\nLet's try other classifier.")
from sklearn.ensemble import VotingClassifier

clf1 = DecisionTreeClassifier(min_samples_split=2)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2), ('gnb', clf3)], 
                                    voting='soft')

params = {'dt__min_samples_split': [2, 3, 4, 5], 
          'rf__n_estimators': [20, 30, 50],
          }

grid = GridSearchCV(estimator=eclf, param_grid=params, cv=sss, scoring='f1')
grid = grid.fit(features, labels)

grid_prediction = grid.predict(features_test)
grid_report = classification_report(grid_prediction, labels_test)
print(grid_report)

print("Best score: %0.3f" % grid.best_score_)
print("Best parameters set:")
grid_best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(params.keys()):
    print("\t%s: %r" % (param_name, grid_best_parameters[param_name]))
    
# Classifiers tester
test_classifier(eclf, my_dataset, new_features_list)