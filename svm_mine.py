import numpy as np 
import pandas as pd 
train_data= pd.read_csv(r'risk_analytics_train.csv',index_col=0,header=0)
test_data=pd.read_csv(r'risk_analytics_test.csv',index_col=0,header=0)
print(train_data.shape)
print(test_data.shape)
train_data.head()
test_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
for value in["Gender","Married","Self_Employed"]:
    train_data[value].fillna(train_data[value].mode()[0],inplace=True)
    
for value in["Dependents","LoanAmount","Loan_Amount_Term","Credit_History"]:
    train_data[value].fillna(train_data[value].median(),inplace=True)
    
train_data.isnull().sum()
for value in["Gender","Self_Employed",]:
    test_data[value].fillna(test_data[value].mode()[0],inplace=True)
for value in["Dependents","Credit_History","LoanAmount","Loan_Amount_Term"]:
    test_data[value].fillna(test_data[value].median(),inplace=True)
test_data.isnull().sum()
median_values = train_data[["Dependents", "LoanAmount", "Loan_Amount_Term", "Credit_History"]].median()
print(median_values)
mode_values = train_data[["Dependents", "LoanAmount", "Loan_Amount_Term", "Credit_History"]].mode()
print(mode_values)
#transforming Categorical data to numerical
from sklearn.preprocessing import LabelEncoder

colname = ["Gender","Married","Education","Self_Employed","Property_Area","Loan_Status"]

le=LabelEncoder()

for x in colname:
    train_data[x]=le.fit_transform(train_data[x])
train_data.head()
#transforming Categorical data to numerical
from sklearn.preprocessing import LabelEncoder

colname = ["Gender","Married","Education","Self_Employed","Property_Area"]

le=LabelEncoder()

for x in colname:
    test_data[x]=le.fit_transform(test_data[x])
test_data.head()

X_train=train_data.values[:,0:-1]
Y_train=train_data.values[:,-1]
Y_train=Y_train.astype(int)
X_train.shape

X_test = test_data.values[:,:] 
X_test.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

print(X_train)
print(X_test)
from sklearn.svm import SVC
svc_model = SVC(kernel='rbf',C=20,gamma=0.01)
svc_model.fit(X_train,Y_train)
Y_pred = svc_model.predict(X_test)
print(list(Y_pred))
svc_model.score(X_train,Y_train)

test_data = pd.read_csv(r'risk_analytics_test.csv',header=0)
test_data["Y_predictions"]=Y_pred
test_data.head()
test_data["Y_predictions"]=test_data["Y_predictions"].map({1:"Eligible",0:"Not Eligible"})
test_data.head()
test_data.to_csv(r'test_data_output1.csv',index=False)
test_data.Y_predictions.value_counts()

