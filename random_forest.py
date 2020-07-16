#Pandas is used for data manipulation
import pandas as pd
#Numpy is requested by another packages to convert to arrays
import numpy as np
#Skicit-learn is used to split data into training and testing sets
from sklearn.model_selection import train_test_split
#Import the model to be used
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Read in data
df = pd.read_csv("C:\\Users\\Pedro\\Documents\\Códigos\\FSI\\Random_Forest_FlowerRecognition\\leaf.csv", header = 0)

# Labels are the values we want to predict
labels = np.array(df['H1'])

# Remove the labels from the features
df = df.drop('H1', axis = 1)
df = df.drop('H2', axis = 1)

#Convert to numpy array
df = np.array(df)

#Split data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df, labels, test_size = 0.25, random_state = 8)

#Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 8)

# Train the model on training data
rf.fit(train_features, train_labels)

#Prediction is next
test_prediction = rf.predict(test_features)

#Now evaluating the model
print(confusion_matrix(test_labels,test_prediction))
print(classification_report(test_labels,test_prediction))
print(accuracy_score(test_labels,test_prediction))
