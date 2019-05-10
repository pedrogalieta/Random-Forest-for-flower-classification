# Pandas is used for data manipulation
import pandas as pd
# Use numpy to convert to arrays
import numpy as np
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Read in data
df = pd.read_csv('leaf.csv', header = 0)
# Labels are the values we want to predict
labels = np.array(df['H1'])
# Remove the labels from the features
df = df.drop('H1', axis = 1)
df = df.drop('H2', axis = 1)
# Saving feature names for later use
feature_list = list(df.columns)
# Convert to numpy array
df = np.array(df)
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df, labels, test_size = 0.25, random_state = 8)
#print('Training features: ', train_features.shape)
#print('Training labels: ', train_labels.shape)
#print('Testing features: ', test_features.shape)
#print('Testing labels: ', test_labels.shape)

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 8)
# Train the model on training data
rf.fit(train_features, train_labels)
#Prediction is next
test_prediction = rf.predict(test_features)
#Now evaluating the model
print(confusion_matrix(test_labels,test_prediction))
print(classification_report(test_labels,test_prediction))
print(accuracy_score(test_labels,test_prediction))
