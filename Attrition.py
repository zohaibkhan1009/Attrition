#Importing the Dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import os
import pylab as pl
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")

data = pd.read_csv("Attrition.csv")

data.head()

pd.set_option('display.max_columns', None)

data.head()

data.isnull().sum()


data[data.duplicated()]

data.drop_duplicates(inplace=True)


def zohaib (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data

data.boxplot(column=["Age"])

data = zohaib(data,"Age")


data.boxplot(column=["DailyRate"])

data = zohaib(data,"DailyRate")


data.boxplot(column=["DistanceFromHome"])

data = zohaib(data,"DistanceFromHome")


data.boxplot(column=["Education"])

data = zohaib(data,"Education")

data.boxplot(column=["EnvironmentSatisfaction"])

data = zohaib(data,"EnvironmentSatisfaction")

data.boxplot(column=["HourlyRate"])

data = zohaib(data,"HourlyRate")

data.boxplot(column=["MonthlyIncome"])

data = zohaib(data,"MonthlyIncome")

data.boxplot(column=["JobSatisfaction"])

data = zohaib(data,"JobSatisfaction")

data.boxplot(column=["JobInvolvement"])

data = zohaib(data,"JobInvolvement")

data.boxplot(column=["NumCompaniesWorked"])

data = zohaib(data,"NumCompaniesWorked")


data.boxplot(column=["MonthlyRate"])

data = zohaib(data,"MonthlyRate")


data.boxplot(column=["TotalWorkingYears"])

data = zohaib(data,"TotalWorkingYears")

data.boxplot(column=["YearsAtCompany"])

data = zohaib(data,"YearsAtCompany")

data.boxplot(column=["YearsWithCurrManager"])

data = zohaib(data,"YearsWithCurrManager")

#Label Encoding
from sklearn import preprocessing
for col in data.select_dtypes(include=['object']).columns:
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(data[col].unique())
    data[col] = label_encoder.transform(data[col])
    print(f'{col} : {data[col].unique()}')


data.info()

#Auto EDA

Autoviz will work on CSV file format.
#first we need to install the autoviz libarary.

from autoviz.AutoViz_Class import AutoViz_Class 

AV = AutoViz_Class()

import matplotlib.pyplot as plt
%matplotlib INLINE
filename = 'Attrition.csv'
sep =","
dft = AV.AutoViz(
    filename  
)

def class_distribution(data, column_name='Attrition'):
    # Display total counts and percentage for each class
    distribution = data[column_name].value_counts()
    percentage = data[column_name].value_counts(normalize=True) * 100
    
    print(f"Class distribution for '{column_name}':")
    print(distribution)
    print("\nPercentage distribution:")
    print(percentage)

# Call the function to display the distribution for the 'Resigned' column
class_distribution(data, 'Attrition')


X = data.drop("Attrition", axis = 1)

y = data["Attrition"]

X.head()

y.head()

#Training and Testing the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

#Scaling the Dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
smote = SMOTE(random_state=42) 
X_resampled,y_resampled = smote.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.3, random_state = 20)



from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

print("Before SMOTE: ", X_train.shape, y_train.shape)
print("After SMOTE: ", X_train_over.shape, y_train_over.shape)
print("After SMOTE Label Distribution: ", pd.Series(y_train_over).value_counts())













#Running the algorith
#Logistic Regression
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)

classifier.fit(X_train,y_train)


#Testing the model

y_predict_test = classifier.predict(X_test)



#Checking confusion matrix and accuracy of the model

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_predict_test,y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict_test,))

from sklearn.metrics import accuracy_score
predictions = classifier.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(str(np.round(acc*100, 2))+'%')



#Random Forest
n_est=100
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=n_est, n_jobs=-1, random_state=0)
rfc.fit(X_train, y_train);

print("Training data accuracy:", rfc.score(X_train, y_train))
print("Testing data accuracy", rfc.score(X_test, y_test)) 

from sklearn.metrics import accuracy_score
predictions = rfc.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(str(np.round(acc*100, 2))+'%')

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_predict_test,y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict_test,))

from sklearn.metrics import accuracy_score
predictions = rfc.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(str(np.round(acc*100, 2))+'%')



#XGBOOST Classifier with accuracy score

from xgboost import XGBClassifier

xgbc = XGBClassifier(tree_method='auto', n_estimators=n_est, n_jobs=-1, random_state=0)
xgbc.fit(X_train, y_train);

print("Training data accuracy:", xgbc.score(X_train, y_train))
print("Testing data accuracy", xgbc.score(X_test, y_test))

from sklearn.metrics import accuracy_score
predictions = xgbc.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(str(np.round(acc*100, 2))+'%')



#HYPER PARA TUNING
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 150, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0, 0.01, 0.1, 1]
}

# Initialize the model
xgb_model = XGBClassifier(tree_method='auto', n_jobs=-1, random_state=0)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,  # Number of random combinations to try; increase for better results
    scoring='f1_weighted',  # Target a balanced F1 score
    cv=5,
    verbose=1,
    random_state=0,
    n_jobs=-1
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best model after hyperparameter tuning
best_model = random_search.best_estimator_

# Model evaluation on training and test data
print("Training data accuracy:", best_model.score(X_train, y_train))
print("Testing data accuracy:", best_model.score(X_test, y_test))

# Predictions on test data
y_pred = best_model.predict(X_test)

# Evaluation metrics
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Detailed accuracy, precision, recall, and F1 scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nAccuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")

# Print the best parameters found
print("\nBest Hyperparameters:")
print(random_search.best_params_)
