
# preprocessing.py: Data Preprocessing for Fraud Detection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv('../dataset/credit_card_transactions.csv')

# Drop any rows with missing values
df_clean = df.dropna()

# Convert categorical variables using One-Hot Encoding
df_clean = pd.get_dummies(df_clean, columns=['merchant', 'category', 'customer_demographics'])

# Separate features and target variable
X = df_clean.drop(columns=['is_fraud'])
y = df_clean['is_fraud']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
