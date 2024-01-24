# Import pandas for database
import pandas as pd

# Import scikit-learn for AI training
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

# Pickle is used to save and load the model
import pickle


# Load the model from disk
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
cv = pickle.load(open("cv.sav", "rb"))

# Load test email
final_predict = cv.transform(["""

C0NGRATULATIONS ****@gmail.com !
A.balance..0F $1000.00 ls AVAlLABLE F0R..your *CashApp*.Accountt
Thiss.TRANSACTlON.may.0nly.appearr. 0n.your.ACC0UNTT..afterr.VALlDATE.your.lnfo.

FUNDlNG.For: **** EMAlL: ****@gmail.com Balance Amount: $1000.00 	
PAY0UT:
$1000.000
Confirm Here
Memo PAY0UT	SlGNATURE ****
"""])
X_train = final_predict
pred = loaded_model.predict(X_train)
print(pred)