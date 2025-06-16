import pandas as pd
from sqlalchemy import create_engine

# MySQL DB credentials
username = "root"
password = "MYSQL"  # 👈 Replace with your actual MySQL password
host = "localhost"
database = "superstore"

# Connect to MySQL
engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}/{database}")

# Read the SQL file (ensure the path is correct)
try:
    with open("sql/retail_analysis.sql", "r", encoding='utf-8') as file:
        lines = file.readlines()
    print("✅ SQL file loaded successfully.")
except Exception as e:
    print(f"❌ Error loading SQL file: {e}")
    exit()

# Combine lines, remove comments, and extract SELECT queries
queries = []
current_query = ""

for line in lines:
    stripped = line.strip()

    if stripped.startswith("--") or not stripped:
        continue  # Skip comments and blank lines

    current_query += " " + stripped

    if ";" in stripped:  # Query ends
        if "select" in current_query.lower():
            queries.append(current_query.strip().rstrip(";"))
        current_query = ""

if not queries:
    print("⚠️ No SELECT queries found in SQL file.")
    exit()

print(f"🔍 Found {len(queries)} SELECT queries.\n")

# Execute each query and display result
for i, query in enumerate(queries, start=1):
    print(f"\n📌 Query {i}:\n{query}\n")
    try:
        df = pd.read_sql(query, con=engine)
        print(df.head(10))
    except Exception as e:
        print(f"❌ Error running Query {i}: {e}")
