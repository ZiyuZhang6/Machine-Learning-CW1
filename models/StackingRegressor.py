import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load train data
train_df = pd.read_csv('processed_train.csv')
X_train = train_df.drop(columns=['outcome'])
y_train = train_df['outcome']

# 80% for training, 20% for validation
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

# Load test data
test_df = pd.read_csv('processed_test.csv')

# Define base model
base_models = [
    ('cat', CatBoostRegressor(iterations=500, learning_rate=0.028044757781398595, depth=6, verbose=100,)),
]

# Define meta model
meta_model = Ridge(alpha=1.0)

# Create StackingRegressor
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, n_jobs=-1)

# Train stacking model
stacking_model.fit(X_train_final, y_train_final)

# Calculate R2 score based on train dataset
y_train_pred = stacking_model.predict(X_val)
train_r2 = r2_score(y_val, y_train_pred)
print(f"Training Set R² Score: {train_r2:.6f}")

# Test set predictions
yhat_lm = stacking_model.predict(test_df)
out = pd.DataFrame({'yhat': yhat_lm})
out.to_csv('CW1_submission_K23069561.csv', index=False) 
