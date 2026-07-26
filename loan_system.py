"""CREDITWISE LOAN SYSTEM - SUPERVISED ML"""

import pandas as pd
import numpy as np


loan_sys = pd.read_csv(r"C:\Users\arsha\Desktop\prime ai&ml\project\loan_approval_data.csv")
pd.set_option("display.max_rows" , None)
pd.set_option("display.max_columns" , None)
# print(loan_sys.head())
# print(loan_sys.isnull().sum())
#handle missing values
categorical_col = loan_sys.select_dtypes(include = ["object"]).columns
print(categorical_col)
numerical_col = loan_sys.select_dtypes(include = ["int64" , "float64"]).columns
print(numerical_col)

from sklearn.impute import SimpleImputer

numerical_imputer = SimpleImputer(strategy="mean")
loan_sys[numerical_col] = numerical_imputer.fit_transform(loan_sys[numerical_col])
# print(loan_sys.head())
# print(loan_sys.isnull().sum())
categotical_imputer = SimpleImputer(strategy="most_frequent")
loan_sys[categorical_col] = categotical_imputer.fit_transform(loan_sys[categorical_col])
print(loan_sys.head())
print(loan_sys.isnull().sum())

#EDA - exploratory data analysis
# how balance our classes is

import matplotlib.pyplot as plt
import seaborn as sns


"""classes_count = loan_sys["Loan_Approved"].value_counts()
plt.pie(classes_count , labels=["no" , "yes"] , autopct="%1.1f%%")
plt.title("is loan approved or not")"""
#plt.show()

#analyze categories
"""gender_cnt = loan_sys["Gender"].value_counts()
ax = sns.barplot(gender_cnt)
ax.bar_label(ax.containers[0]) """
# plt.show()


"""emp_status = loan_sys["Employment_Status"].value_counts()
ax = sns.barplot(emp_status)
ax.bar_label(ax.containers[0])"""
# plt.show()

#analyze annaul income 
"""sns.histplot(
    data=loan_sys,
    x="Applicant_Income",
    bins=20
)"""
# plt.show()

#outliers - boxplots
"""
fig , axes = plt.subplots(2,2)

sns.boxplot(ax = axes[0,0] , data = loan_sys , x="Loan_Approved" , y = "Applicant_Income")
sns.boxplot(ax = axes[0,1] , data = loan_sys , x="Loan_Approved" , y = "Credit_Score")
sns.boxplot(ax = axes[1,0] , data = loan_sys , x="Loan_Approved" , y = "DTI_Ratio")
sns.boxplot(ax = axes[1,1] , data = loan_sys , x="Loan_Approved" , y = "Savings")
plt.tight_layout( )"""
# plt.show()


#check credit scoree

"""sns.histplot(
    data = loan_sys,
    x = "Credit_Score",
    hue = "Loan_Approved",
    bins=20,
    multiple="dodge
)"""
# plt.show()

#remove applicant id
loan_sys = loan_sys.drop("Applicant_ID" , axis=1)
print(loan_sys)
print(loan_sys.info())


from sklearn.preprocessing import LabelEncoder , OneHotEncoder

le = LabelEncoder()
loan_sys["Education_Level"] = le.fit_transform(loan_sys["Education_Level"])
loan_sys["Loan_Approved"] = le.fit_transform(loan_sys["Loan_Approved"])

# print(loan_sys.head())

#for encoder  - srf object wale column leskte h
cols = ["Employment_Status" , "Marital_Status" , "Loan_Purpose" , "Property_Area" , "Gender" , "Employer_Category"]
ohe = OneHotEncoder(drop="first" , sparse_output=False , handle_unknown="ignore")
encoded = ohe.fit_transform(loan_sys[cols])
#convert to dataframe
encoded_loan_sys = pd.DataFrame(encoded , columns=ohe.get_feature_names_out(cols) , index = loan_sys.index)
# print(encoded_loan_sys.head())

loan_sys = pd.concat([loan_sys.drop(columns = cols) , encoded_loan_sys] , axis = 1)
print(loan_sys.head()) 
print(loan_sys.info())

#correlation heatmap

nums_cols = loan_sys.select_dtypes(include = "number")
corr_matrix = nums_cols.corr()
# print(corr_matrix)
print(nums_cols.corr()["Loan_Approved"].sort_values(ascending=False))

plt.figure(figsize=(15,8))
sns.heatmap(
    corr_matrix,
    fmt="0.2f",
    annot=True,
    cmap="coolwarm"
)
# plt.show()
# plt.tight_layout()


#  FEATURE SCALING
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = loan_sys.drop(columns = ["Loan_Approved"] , axis = 1)
y = loan_sys["Loan_Approved"]

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size=0.2 , random_state=42
)
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)

print(X_test_scaled)

"""TRAIN AND EVALUATE MODELS"""
#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score , recall_score , f1_score ,accuracy_score , confusion_matrix
log_model = LogisticRegression()
log_model.fit(X_train_scaled , y_train)
y_pred = log_model.predict(X_test_scaled)

#EVALUATION
print("LOGISTIC REGRESSION MODEL")
print("precision score : " ,precision_score(y_test , y_pred))
print("recall score : " , recall_score(y_test , y_pred))
print("f1_score : " , f1_score(y_test , y_pred))
print("accuracy score : " ,accuracy_score(y_test , y_pred))
print("confusion matrix : " , confusion_matrix(y_test , y_pred) )

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled , y_train)
y_pred = knn_model.predict(X_test_scaled)

#EVALUATION
print("KNN MODEL")
print("precision score : " ,precision_score(y_test , y_pred))
print("recall score : " , recall_score(y_test , y_pred))
print("f1_score : " , f1_score(y_test , y_pred))
print("accuracy score : " ,accuracy_score(y_test , y_pred))
print("confusion matrix : " , confusion_matrix(y_test , y_pred) )

#NAIVE BYES
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled , y_train)
y_pred = gnb_model.predict(X_test_scaled)

#EVALUATION
print("NAIVE BAYES MODEL")
print("precision score : " ,precision_score(y_test , y_pred))
print("recall score : " , recall_score(y_test , y_pred))
print("f1_score : " , f1_score(y_test , y_pred))
print("accuracy score : " ,accuracy_score(y_test , y_pred))
print("confusion matrix : " , confusion_matrix(y_test , y_pred) )

"""BEST MODEL ON THE BASIS OF PRECISION IS NAIVE BAYES MODEL"""

"""FEATURE ENGIEERING"""

#Add or transform features
loan_sys["DTI_Ratio_sq"] = loan_sys["DTI_Ratio"] ** 2
loan_sys["Credit_Score_sq"] = loan_sys["Credit_Score"] ** 2

# loan_sys["Application_Income_log"] = np.log1p(loan_sys["Application_Income"])

X = loan_sys.drop(columns = ["Loan_Approved" , "Credit_Score" , "DTI_Ratio" ])
y = loan_sys["Loan_Approved"]

#train test split
X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size=0.2 , random_state=42
)
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)
# print(X_train.head())

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score , recall_score , f1_score ,accuracy_score , confusion_matrix
log_model = LogisticRegression()
log_model.fit(X_train_scaled , y_train)
y_pred = log_model.predict(X_test_scaled)

#EVALUATION
print("LOGISTIC REGRESSION MODEL AFTER FEATURE ENGINEERING")
print("precision score : " ,precision_score(y_test , y_pred))
print("recall score : " , recall_score(y_test , y_pred))
print("f1_score : " , f1_score(y_test , y_pred))
print("accuracy score : " ,accuracy_score(y_test , y_pred))
print("confusion matrix : " , confusion_matrix(y_test , y_pred) )

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled , y_train)
y_pred = knn_model.predict(X_test_scaled)

#EVALUATION
print("KNN MODEL AFTER FEATURE ENGINEERING")
print("precision score : " ,precision_score(y_test , y_pred))
print("recall score : " , recall_score(y_test , y_pred))
print("f1_score : " , f1_score(y_test , y_pred))
print("accuracy score : " ,accuracy_score(y_test , y_pred))
print("confusion matrix : " , confusion_matrix(y_test , y_pred) )

#NAIVE BYES
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled , y_train)
y_pred = gnb_model.predict(X_test_scaled)

#EVALUATION
print("NAIVE BAYES MODEL AFTER FEATURE ENGINEERING")
print("precision score : " ,precision_score(y_test , y_pred))
print("recall score : " , recall_score(y_test , y_pred))
print("f1_score : " , f1_score(y_test , y_pred))
print("accuracy score : " ,accuracy_score(y_test , y_pred))
print("confusion matrix : " , confusion_matrix(y_test , y_pred) )

"""AFTER FEATURE ENGINEERING BEST MODEL PERFORM IS NAIVE BYES"""
