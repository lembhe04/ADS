import pandas as pd

file_path = r"C:\Users\ADMIN\Desktop\python\mental_health_digital_behavior_time.csv"  
# 👆 change this path to your actual CSV location

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully ✅")
print(df.head())
print("\nColumns:\n", df.columns)
print("\nShape:", df.shape)
