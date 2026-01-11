import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")

X = data[['Advertising']]
y = data['Sales']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

print("Model Score:", model.score(X_test, y_test))

# Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Advertising Spend")
plt.ylabel("Sales")
plt.show()
