# Import pandas for database
import pandas as pd

# Import scikit-learn for AI training
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

# Pickle is used to save and load the model
import pickle

# Read the training data
spam = pd.read_csv(r"spam.csv")
# Label of spam vs ham (good)
z = spam["v2"]

# Email Text
y = spam["v1"]
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2)

# Convert the text into a vector
cv = CountVectorizer()
features = cv.fit_transform(z_train)

# Create a model and train it on the data
model = svm.SVC()
model.fit(features,y_train)

# Test the model and print accuracy
features_test = cv.transform(z_test)
print(model.score(features_test,y_test))

# Save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Save cv features for vectorization
pickle.dump(cv, open("cv.sav", "wb"))
 
