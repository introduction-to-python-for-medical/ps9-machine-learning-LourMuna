import pandas as pd
df = pd.read_csv('/parkinsons.csv')
print(df.head())
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df)
plt.show()
input_features = ['MDVP:Fo(Hz)', 'MDVP:Flo(Hz)']
output_feature = 'status'
print("Input features:", input_features)
print("Output feature:", output_feature)
# print(idk)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[input_features] = scaler.fit_transform(df[input_features])
print(df.head())
# print(idk)
# prompt: Divide the dataset into a training set and a validation set.

from sklearn.model_selection import train_test_split

# Assuming 'df' is your DataFrame and 'input_features', 'output_feature' are defined
X = df[input_features]
y = df[output_feature]

# Split the data into training and validation sets (e.g., 80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) #random_state for reproducibility

from sklearn.linear_model import LogisticRegression

# Initialize the model (Logistic Regression in this case, as suggested by the paper)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")

if accuracy < 0.8:
    print("Accuracy is below the target of 0.8. Consider trying different features, models, or hyperparameters.")
import joblib
joblib.dump(knn_model, 'DecisionTreeClassifier.joblib')
