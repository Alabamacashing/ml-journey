from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(df.head())
print(df.shape)
print(df.isnull().sum())

cluster_features = df[["MedInc", "HouseAge"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

features = df[["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]]
target = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)

mae = mean_absolute_error(y_test, predictions)
print(mae)
print(df["MedHouseVal"].mean())

df["cluster"] = kmeans.labels_
print(df["cluster"].value_counts())
print(df.groupby("cluster")[["MedInc", "HouseAge"]].mean())