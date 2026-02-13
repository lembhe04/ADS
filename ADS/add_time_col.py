import pandas as pd
import numpy as np

# Load dataset
file_path = r"C:\Users\ADMIN\Desktop\python\mental_health_data.csv"
df = pd.read_csv(file_path)

n = len(df)

# Define start and end dates
start_date = pd.to_datetime("2020-01-01")
end_date   = pd.to_datetime("2024-12-31")

# Create smooth continuous date sequence
date_range = pd.to_datetime(
    np.linspace(start_date.value, end_date.value, n)
)

# Add columns
df["date"] = date_range
df["year"] = df["date"].dt.year      # ✅ extract year first
df["date"] = df["date"].dt.date      # ✅ then clean date format

print("Smooth continuous time added ✅")
print(df[["date", "year"]].head(10))
print(df[["date", "year"]].tail(10))

# Save new dataset
new_file_path = r"C:\Users\ADMIN\Desktop\python\mental_health_digital_behavior_time.csv"
df.to_csv(new_file_path, index=False)

print("\nSaved as:", new_file_path)
