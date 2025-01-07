import pandas as pd
df = pd.read_csv('/content/parkinsons.csv')
selected_features = ['D2', 'PPE']
output_feature = 'status'
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
X = df[selected_features]
y = df[output_feature]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_scaled, y)
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
