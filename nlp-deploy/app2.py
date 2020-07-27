import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


df= pd.read_csv("spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
joblib.dump(clf, 'Model.pkl')
NB_spam_model = open('Model.pkl','rb')
clf = joblib.load(NB_spam_model)

message = "Click here to get a million dollor offer"
data = [message]
vect = cv.transform(data).toarray()
my_prediction = clf.predict([['Click here to get a million dollor offer']])
print(my_prediction)
print("The Test..")

	

	
	
	

	
	# Features and Labels
	