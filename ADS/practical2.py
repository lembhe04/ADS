import pandas as pd
import numpy as np

# Load dataset with time column
file_path = r"C:\Users\ADMIN\Desktop\python\mental_health_digital_behavior_time.csv"
df = pd.read_csv(file_path)

# Select variable
col = "daily_screen_time_min"

print("\n==============================")
print("PRACTICAL 2 : STATISTICAL ANALYSIS")
print("==============================")

# -------------------------------
# PART 1 : UNGROUPED DATA
# -------------------------------
print("\n--- PART 1 : UNGROUPED DATA ---")

mean_ungrouped = df[col].mean()
median_ungrouped = df[col].median()
mode_ungrouped = df[col].mode()[0]
var_ungrouped = df[col].var()
std_ungrouped = df[col].std()

print(f"Mean (Ungrouped)   : {mean_ungrouped:.2f}")
print(f"Median (Ungrouped) : {median_ungrouped:.2f}")
print(f"Mode (Ungrouped)   : {mode_ungrouped:.2f}")
print(f"Variance (Ungrouped): {var_ungrouped:.2f}")
print(f"Std Dev (Ungrouped): {std_ungrouped:.2f}")

# -------------------------------
# PART 2 : GROUPED DATA
# -------------------------------
print("\n--- PART 2 : GROUPED DATA ---")

# Create class intervals (bins)
bins = np.linspace(df[col].min(), df[col].max(), 6)  # 5 groups
df["group"] = pd.cut(df[col], bins=bins)

# Frequency table
grouped_freq = df.groupby("group", observed=False)[col].count().reset_index(name="frequency")

# Class midpoints
grouped_freq["midpoint"] = grouped_freq["group"].apply(lambda x: (x.left + x.right)/2)

# 🔥 Convert midpoint to numeric
grouped_freq["midpoint"] = grouped_freq["midpoint"].astype(float)

# Grouped mean
grouped_mean = (grouped_freq["midpoint"] * grouped_freq["frequency"]).sum() / grouped_freq["frequency"].sum()

# Grouped variance
grouped_var = (grouped_freq["frequency"] * (grouped_freq["midpoint"] - grouped_mean)**2).sum() / grouped_freq["frequency"].sum()

# Grouped std dev
grouped_std = np.sqrt(grouped_var)

print("\nGrouped Frequency Table:")
print(grouped_freq)

# -------------------------------
# GROUPED MODE CALCULATION
# -------------------------------

# Find modal class (highest frequency)
modal_row = grouped_freq.loc[grouped_freq["frequency"].idxmax()]

modal_class = modal_row["group"]
f1 = modal_row["frequency"]

# Get index of modal class
idx = grouped_freq.index[grouped_freq["group"] == modal_class][0]

# f0 and f2
f0 = grouped_freq.iloc[idx-1]["frequency"] if idx > 0 else 0
f2 = grouped_freq.iloc[idx+1]["frequency"] if idx < len(grouped_freq)-1 else 0

# Class boundaries
L = modal_class.left      # lower boundary
h = modal_class.right - modal_class.left   # class width

# Grouped mode formula
grouped_mode = L + ((f1 - f0) / (2*f1 - f0 - f2)) * h

print(f"\nMode (Grouped) : {grouped_mode:.2f}")

# -------------------------------
# GROUPED MEDIAN CALCULATION
# -------------------------------

# Total frequency
N = grouped_freq["frequency"].sum()

# Cumulative frequency
grouped_freq["cf"] = grouped_freq["frequency"].cumsum()

# Find median class
median_class_row = grouped_freq[grouped_freq["cf"] >= N/2].iloc[0]

median_class = median_class_row["group"]
f = median_class_row["frequency"]
cf_prev = grouped_freq.loc[grouped_freq.index[grouped_freq["group"] == median_class][0] - 1, "cf"] \
          if grouped_freq.index[grouped_freq["group"] == median_class][0] > 0 else 0

# Class boundaries
L = median_class.left
h = median_class.right - median_class.left

# Grouped median formula
grouped_median = L + ((N/2 - cf_prev) / f) * h
print(f"\nMedian (Grouped) : {grouped_median:.2f}")
print(f"\nMean (Grouped)     : {grouped_mean:.2f}")
print(f"Variance (Grouped) : {grouped_var:.2f}")
print(f"Std Dev (Grouped)  : {grouped_std:.2f}")
