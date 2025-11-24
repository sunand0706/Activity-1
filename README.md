import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

data = {
    'Appliances': [6, 8, 10, 12, 14, 16, 18, 20],
    'Members': [2, 3, 4, 4, 5, 5, 6, 7],
    'Usage_Hours': [4, 5, 5, 6, 6, 7, 7, 8],
    'Temperature': [25, 28, 29, 30, 32, 33, 34, 36],
    'Unit_Consumption': [160, 200, 260, 310, 350, 390, 440, 490]
}

df = pd.DataFrame(data)

X = df[['Appliances', 'Members', 'Usage_Hours', 'Temperature']]
y = df['Unit_Consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Predicted Units:", pred)
print("RMSE:", rmse)
