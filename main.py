import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('/parkinsons.csv')
print(df.head())

# Visualize data distribution
sns.pairplot(df)
plt.show()

# Input and Output features
input_features = ['MDVP:Fo(Hz)', 'MDVP:Flo(Hz)']
output_feature = 'status'
print("Input features:", input_features)
print("Output feature:", output_feature)

# Normalize the input features
scaler = MinMaxScaler()
df[input_features] = scaler.fit_transform(df[input_features])
print(df.head())

# Split the dataset into training and validation sets
X = df[input_features]
y = df[output_feature]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")

# If accuracy is below 0.8, print a suggestion for improvement
if accuracy < 0.8:
    print("Accuracy is below the target of 0.8. Consider trying different features, models, or hyperparameters.")

# Save the trained model
joblib.dump(model, 'LogisticRegressionModel.joblib')
