
"""
Created on Sunday October 12, 2025, 20:38:27

@author: Tanmay Basu
"""

import csv,os,re,sys,codecs
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib,  statistics
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter


# Statistics of individual classes
def get_class_statistics(labels):
    class_statistics=Counter(labels)
    print('\n Class \t\t Number of Instances \n')
    for item in list(class_statistics.keys()):
        print('\t'+str(item)+'\t\t\t'+str(class_statistics[item]))

path = '/home/Downloads/23060_Arnav_Kapoor/assign1/code/'

# Allow passing a path as the first CLI argument
if len(sys.argv) > 1:
  path = sys.argv[1]

# Resolve the file path robustly
def resolve_wine_csv(p: str) -> Path:
  p = Path(p)
  # if a directory was provided, look for winequality_white.csv inside it
  if p.is_dir():
    candidate = p / 'winequality_white.csv'
    if candidate.exists():
      return candidate
  # if a path to a file was provided
  if p.is_file() and p.suffix.lower() == '.csv':
    return p
  # try relative to script folder
  candidate = Path(__file__).resolve().parent / 'winequality_white.csv'
  if candidate.exists():
    return candidate
  # try current working directory
  candidate = Path.cwd() / 'winequality_white.csv'
  if candidate.exists():
    return candidate
  # last attempt: if p doesn't end with .csv, append filename
  if p.exists():
    candidate = p / 'winequality_white.csv'
    if candidate.exists():
      return candidate
  raise FileNotFoundError(f"Could not find 'winequality_white.csv'. Looked in: {p}, {Path(__file__).resolve().parent}, {Path.cwd()}")
 
# Load the file using Pandas       
csv_path = resolve_wine_csv(path)
reader = pd.read_csv(csv_path)


# mask = reader['target'] == 9
# reader = reader[~mask]

# Detect label column (common names) or fall back to last column
label_col = None
for col in ('target', 'quality', 'label', 'Product Quality', 'class'):
  if col in reader.columns:
    label_col = col
    break

if label_col is not None:
  labels = reader[label_col]
  data = reader.drop(columns=[label_col])
else:
  # assume last column is the label
  labels = reader.iloc[:, -1]
  data = reader.iloc[:, :-1]

get_class_statistics(labels)
print('\nProceeding with training/validation split...')
     
# Training and test split WITHOUT stratification        
# training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels, 
#                                                test_size=0.10, random_state=42)

# Training and test split WITH stratification   
training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels, 
                                               test_size=0.5, random_state=42,stratify=labels)
print('\n Training Data ')
training_cat=[str(x) for x in training_cat]
get_class_statistics(training_cat)
print('\nProceeding to training...')

print('\n Validation Data ')
validation_cat=[str(x) for x in validation_cat]
get_class_statistics(labels)
print('\nProceeding to classification...')

  # Classification
     
clf1 = LogisticRegression(solver='liblinear',class_weight='balanced') 
clf2 = RandomForestClassifier(max_features=None,class_weight='balanced')
clf3 = svm.SVC(class_weight='balanced',kernel='linear',C=1,probability=True)
clf3 = MultinomialNB(fit_prior=True, class_prior=None)
clf4 = DecisionTreeClassifier(random_state=40) 
clf5 = svm.LinearSVC(class_weight='balanced') 

clf1.fit(training_data,training_cat)
predicted=clf1.predict(validation_data)
class_names=[str(item) for item in list(Counter(validation_cat).keys())]

# Classification report
print('\n ##### Classification Report ##### \n')
print(classification_report(validation_cat, predicted, target_names=class_names))


pr=precision_score(validation_cat, predicted, average='macro') 
print ('\n Precision:\t'+str(pr)) 

rl=recall_score(validation_cat, predicted, average='macro') 
print ('\n Recall:\t'+str(rl))

fm=f1_score(validation_cat, predicted, average='macro') 
print ('\n F1-Score:\t'+str(fm))

