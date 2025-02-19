import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR

# Load train data
train_df = pd.read_csv('processed_train.csv')
X_train = train_df.drop(columns=['outcome'])
y_train = train_df['outcome']

# 80% for training, 20% for validation
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=123),
    "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=123),
    "LightGBM": LGBMRegressor(learning_rate=0.02, max_depth=4, n_estimators=700, num_leaves=16,random_state=123),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=123),
    "CatBoost": CatBoostRegressor(iterations=700, learning_rate=0.02, depth=4, random_state=123, verbose=0),
    "SVR": SVR(kernel='rbf', C=10, gamma='scale'),
    }

# Train models and evaluate R² scores
results = {}
for name, model in models.items():
    model.fit(X_train_final, y_train_final)
    yhat = model.predict(X_val)
    r2 = r2_score(y_val, yhat)
    results[name] = r2    

# Print all results to see which model perform the best
print("\nModel Performance:")
for model, score in results.items():
    print(f"{model}: {score}")
