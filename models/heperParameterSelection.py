import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from skopt import BayesSearchCV

# Load train data
train_df = pd.read_csv('processed_train.csv')
X_train = train_df.drop(columns=['outcome'])
y_train = train_df['outcome']

# 80% for training, 20% for validation
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

# Define BayesSearch parameters
opt = BayesSearchCV(
    estimator=CatBoostRegressor(verbose=0, random_state=123),
    search_spaces={
        'depth': (3, 10),
        'iterations': (500, 5000),
        'learning_rate': (0.01, 0.1)
    },
    n_iter=20,  
    cv=3,       
    scoring='r2',
    n_jobs=-1,
    random_state=123
)

# Train model
opt.fit(X_train_final, y_train_final)

# Show best R² Score and corresponding paremeters
print("Best R² Score:", opt.best_score_)
print("Best Parameters:", opt.best_params_)
