import pandas as pd
df = pd.read_csv('/content/parkinsons.csv')
df = df.dropna()
import seaborn as sns
sns.pairplot(df, hue= "status"  )
selected_feutures = ['spread1', 'PPE']
target = "status"
x = df[selected_feutures]
y = df[target]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
knn_model = DecisionTreeClassifier(max_depth = 3 )
knn_model.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
y_pred = knn_model.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy on the test set: {accuracy}")
import joblib
joblib.dump(knn_model, 'DecisionTreeClassifier.joblib')
