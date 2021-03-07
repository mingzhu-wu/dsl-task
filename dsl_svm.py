#!/usr/bin/env python
# coding: utf-8

# <h2>Mingzhu Wu</h2>
# <h3>Python for distinguish language varieties</h3>
# 
# <p> 
# Data Source: http://ttg.uni-saarland.de/resources/DSLCC/  
# </p>


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score
import pandas as pd
import timeit

start = timeit.default_timer()

# Read training data
df = pd.read_csv("dslcc4/DSL-TRAIN.txt", sep='\t', names=["Text", "Language"])

# Split data into raw features and labels
language_features = df['Text']
language_labels = df['Language']

X_train, X_test, y_train, y_test = train_test_split(language_features, 
                                                    language_labels,
                                                    test_size = 0.2,
                                                    random_state = 42)

# Make Machine Learning Pipeline with TfidfVectorizer and LinearSVC
tfidf_vect = TfidfVectorizer(analyzer='char', ngram_range=(3,3))
model = LinearSVC()
text_clf = Pipeline([('tfidf', tfidf_vect),
                    ('clf', model),
                    ])

# Training
text_clf.fit(X_train, y_train)
stop = timeit.default_timer()
print("Training time is " + str(stop - start) + " seconds")

# Evalute on DSLCCv4.0 test data
test_data = pd.read_csv("dslcc4/DSL-TEST-GOLD.txt", sep='\t', header=None)
pred = text_clf.predict(test_data[0])

print("Accuracy on test set is:", accuracy_score(test_data[1],pred))


# Predict on a single sentence
# x = ['De la que -- según el teniente alcalde de Lima, Marco Parra -- hizo la firma Latin Pacific Capital para la municipalidad.']
# x = pd.Series(x)
# text_clf.predict(x)

