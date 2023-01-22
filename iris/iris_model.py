import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle  # To save the model that we are going to train so that we can use it directly in our web app.

data=pd.read_excel('.venv\iris.xls')
# X = feature values, all the columns except the last column
X = data.iloc[:, :-1]
# y = target values, last column of the data frame
y = data.iloc[:, -1]
#Split the data into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()  
model.fit(x_train, y_train)       #Training the model

predictions = model.predict(x_test)  #Test the model
print( classification_report(y_test, predictions) )
print(accuracy_score(y_test, predictions)*100 ,'%')

pickle.dump(model,open('model.pkl','wb'))   #dump() function in pickle and save the model
#Pickle is a useful Python tool that allows you to save your ML models with moel.pkl name
#It helsp to minimise lengthy re-training and allow you to share, commit, and re-load pre-trained ML models
p=model.predict([[5.1,3.5,1.4,0.2]])
print(p[0])