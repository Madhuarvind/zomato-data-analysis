import pandas as pd
import numpy as np

# Use raw string (prefix `r`) and set encoding
df = pd.read_csv(r"E:\Retail Sales Analysis\dataset\Superstore.csv", encoding='latin1')

print(df.head())
print(df.info())
print(df.describe())

print("The dataset contail null values",df.isnull())

# Convert date columns to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Drop unnecessary columns
df.drop(['Row ID'], axis=1, inplace=True)  # Optional: Drop Postal Code if not needed

# Check for duplicates
print(f"Duplicates: {df.duplicated().sum()} rows")

# Remove duplicates
df = df.drop_duplicates()


# Extract Year and Month from Order Date
df['Order Year'] = df['Order Date'].dt.year
df['Order Month'] = df['Order Date'].dt.month

# Create Profit Margin column
df['Profit Margin'] = round(df['Profit'] / df['Sales'], 2)

# Optional: Sales Category (Low, Medium, High)
def sales_category(sales):
    if sales < 100:
        return 'Low'
    elif sales < 500:
        return 'Medium'
    else:
        return 'High'

df['Sales Category'] = df['Sales'].apply(sales_category)

# Preview after changes
print(df[['Sales', 'Profit', 'Profit Margin', 'Order Year', 'Order Month', 'Sales Category']].head())

# Save cleaned data
df.to_csv('cleaned_superstore.csv', index=False)


# Required installations (run in terminal if not installed)
# pip install sqlalchemy pymysql pandas

from sqlalchemy import create_engine
import pandas as pd

# Load your cleaned CSV file
df = pd.read_csv("cleaned_superstore.csv", encoding='latin1')

# Replace with your actual MySQL credentials
username = "root"        # ✅ your MySQL username
password = "MYSQL"       # ✅ your MySQL password
host = "localhost"       # ✅ or 127.0.0.1
database = "superstore"  # ✅ must be created beforehand

# SQLAlchemy connection string
engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}/{database}")

# Upload the DataFrame to MySQL
df.to_sql(name='sales_data', con=engine, index=False, if_exists='replace')

print("✅ Data uploaded to MySQL table: sales_data")
