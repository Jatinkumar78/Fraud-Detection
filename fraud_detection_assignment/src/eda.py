
# eda.py: Exploratory Data Analysis for Fraud Detection

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv('../dataset/credit_card_transactions.csv')

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Summary statistics of numerical columns
print(df.describe())

# Distribution of transaction amounts
sns.histplot(df['transaction_amount'], bins=50)
plt.title('Transaction Amount Distribution')
plt.show()

# Count plot for fraud vs legitimate
sns.countplot(x='is_fraud', data=df)
plt.title('Fraud vs Legitimate Transactions')
plt.show()
