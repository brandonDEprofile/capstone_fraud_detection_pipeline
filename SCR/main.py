#Improts
##########
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../data/Fraud.csv")
print(df.head())
print(df.info())

#Cleaning
#########
#Drop n/a
df = df.dropna()
#Duplicates
print("Duplicates:", df.duplicated().sum())
# Drop unnecessary columns
df = df.drop(["nameOrig", "nameDest"], axis=1, errors='ignore')
#Check data type
print(df.dtypes)
#Convert
df["amount"] = df["amount"].astype(float)
#Categorical variables(one-hot encoding)
df = pd.get_dummies(df, columns=["type"], drop_first=True)
print(df.head())
#Class Imbalance
print(df["isFraud"].value_counts())

#split data
##########
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

#Traing model
#########
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model training complete")

#Prediction
##########
y_pred = model.predict(X_test)

#Evaluate model
##########
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", cm)

#Decision tree
#########
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("\nDecision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))

#Plot
#########
df['isFraud'].value_counts().plot(kind='bar')
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

#Plot matrix
#########
# Confusion matrix
cm = np.array([[1270828, 76],
               [834, 786]])

# Plot
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Add labels
classes = ["Non-Fraud", "Fraud"]
plt.xticks([0, 1], classes)
plt.yticks([0, 1], classes)

# Add numbers inside boxes
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j]),
                 ha="center", va="center")

plt.tight_layout()
plt.show()

