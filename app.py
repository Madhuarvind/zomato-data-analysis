import pandas as pd

# Load data
file_path = 'E:\zomoto-Data-Analysis\data\Zomato-data-.csv'  # adjust if needed
df = pd.read_csv(file_path)

# Preview data
print("Shape of dataset:", df.shape)
print("Columns:")
print(df.columns.tolist())
df.head()

# Data types
print(df.info())

# Check missing values
print(df.isnull().sum())

def clean_rate(x):
    if isinstance(x, str):
        x = x.strip()
        if x in ['NEW', '-', '']:
            return None
        else:
            return float(x.split('/')[0])
    return None

df['rate_cleaned'] = df['rate'].apply(clean_rate)

# Check cleaned values
print(df[['rate', 'rate_cleaned']].head())

# Summary of cleaned ratings
print(df['rate_cleaned'].describe())


import random

def generate_review(rating):
    if rating >= 4.0:
        return random.choice(['Amazing food!', 'Loved it!', 'Fantastic service.'])
    elif rating >= 3.0:
        return random.choice(['It was okay.', 'Decent experience.', 'Could be better.'])
    else:
        return random.choice(['Not good.', 'Bad service.', 'Won’t come back.'])

df['review_text'] = df['rate_cleaned'].apply(generate_review)


df.to_csv('zomato_cleaned.csv', index=False)
