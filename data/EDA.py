import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load train dataset
train_df = pd.read_csv("CW1_train.csv")

# Encode categorical features
categorical_cols = ['cut', 'color', 'clarity']

encoder = LabelEncoder()
for col in categorical_cols:
    train_df[col] = encoder.fit_transform(train_df[col])

# Visualizing numerical feature distributions
numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12, 8))
train_df[numerical_features].hist(bins=30, figsize=(12, 8))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# Correlation heatmap to analyze feature relationships
corr_matrix = train_df.corr()  
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

# Find features that are highly correalted
threshold = 0.9
corr_matrix = corr_matrix.abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# Excluded those features
train_df = train_df.drop(columns=to_drop)

# One-hot encode categorical variables
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)

# Save the processed dataset
train_df.to_csv("processed_train.csv", index=False)
print("Processed train dataset saved as 'processed_train.csv'.")

# Load test dataset
test_df = pd.read_csv("CW1_test.csv")

# Encode categorical features
categorical_cols = ['cut', 'color', 'clarity']
encoder = LabelEncoder()
for col in categorical_cols:
    test_df[col] = encoder.fit_transform(test_df[col])

test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

test_df.to_csv("processed_test.csv", index=False)
print("Processed test dataset saved as 'processed_test.csv'.")
