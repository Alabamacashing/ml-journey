import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier



df = pd.read_csv("train.csv")
# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.describe())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop("Cabin", axis=1)
print(df.isnull().sum())
# df.groupby("Pclass")["Survived"].mean().plot(kind="bar")
# plt.title("Survival Rate by Passenger Class")
# plt.xlabel("Passenger Class")
# plt.ylabel("Survival Rate")
# plt.show()
# print(df["Age"].mean())
# print(df["Age"].median())
# print(df["Age"].std())
# print(df["Fare"].std())
# correlation = df["Pclass"].corr(df["Survived"])
# print(correlation)
# print(df["Fare"].corr(df["Survived"]))
# import matplotlib.pyplot as plt

# df["Fare"].hist(bins=20)
# plt.title("Distribution of Fare")
# plt.xlabel("Fare")
# plt.ylabel("Number of Passengers")
# plt.show()


features = df[["Pclass", "Age", "Fare", "Sex",]]
target = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)
import joblib

joblib.dump(model, "titanic_model.joblib")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(accuracy)


