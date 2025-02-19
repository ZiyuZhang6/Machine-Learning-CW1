# Machine-Learning-CW1
1. Exploratory data analysis:
Visualizing numerical feature distributions and Correlation heatmap to analyze feature relationships.
Transform features by encoding categorical features.
Excluded features that are highly correalted.(EDA.py)

2. Model selection:
Compare a lot of regression models including Linear Regression, Ridge Regression, Random Forest, XGBoost,LightGBM, Gradient Boosting, CatBoost and SVR,
show each of their r2 score to select the best model which is CatBoost in this case.(modelSelection.py)

3. Model training and evaluation:
Use BayesSearchCV to select the best parameters for CatBoost.(hyperParameterSelection.py)
However, the r2 score still quite low.
So Stacking Model is used to increase the performance, Catboost is used as base model and Rigde Regression is used as meta model.(StackingRegressor.py)
