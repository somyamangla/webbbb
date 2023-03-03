from sys import displayhook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Indian_liver_patient.csv")
df.head()
df.shape 
df.dtypes
df.describe
df.isnull().sum()
df.duplicated().any()
features = df[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']]
X = df[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']]
y = df['Dataset']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
      displayhook(X_test)
      
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)
from sklearn import model_selection, metrics
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, balanced_accuracy_score
prediction = model.predict(X_test) 
prediction
accuracy = metrics.accuracy_score(y_test,prediction)
accuracy
 
import pickle
from sklearn.linear_model import RandomForestClassifier
pickle.dump(RandomForestClassifier, open('livermodel.pkl', 'wb'))
model = pickle.load(open('livermodel.pkl', 'rb'))
print(model)