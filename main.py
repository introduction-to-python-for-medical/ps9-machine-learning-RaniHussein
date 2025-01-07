import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


    df = pd.read_csv('parkinsons.csv')  # Adjust the path if needed
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'parkinsons.csv' was not found. Please check the file path.")

# Display the first 5 rows of the dataset
print(df.head(5))

# Define selected features and target variable
selected_features = ['D2', 'PPE']  # Ensure these columns exist in the dataset
output_feature = 'status'


    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    print("Model trained successfully!")

    # Evaluate the model
    y_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model
    joblib.dump(knn, 'rani_Pd.joblib')
    print("Model saved as 'rani_Pd.joblib'.")
