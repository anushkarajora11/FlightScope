import pandas as pd
import numpy as np

# 1. Load Data
print("Loading datasets...")
df1 = pd.read_csv("Flights_2022_1.csv", low_memory=False)
df2 = pd.read_csv("Flights_2022_2.csv", low_memory=False)
df3 = pd.read_csv("Flights_2022_3.csv", low_memory=False)
df4 = pd.read_csv("Flights_2022_4.csv", low_memory=False)

# 2. Combine Data
df = pd.concat([df1, df2, df3, df4], ignore_index=True)
print(f"Initial Shape: {df.shape}")

# 3. Drop completely empty columns and unnecessary time columns
df.drop(['Year', 'Quarter'], axis=1, inplace=True, errors='ignore')
df.dropna(axis=1, how='all', inplace=True)

# 4. Remove Duplicate Rows
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Dropped {initial_rows - df.shape[0]} duplicate rows.")

# 5. Handle Delay Columns (Fill NaNs with 0 for non-delayed flights)
delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
existing_delay_cols = [col for col in delay_cols if col in df.columns]
if existing_delay_cols:
    df[existing_delay_cols] = df[existing_delay_cols].fillna(0)

# 6. Handle Cancellation Codes
if 'CancellationCode' in df.columns:
    df['CancellationCode'] = df['CancellationCode'].fillna('N/A')

# 7. Filter out invalid/unusable rows
# If a flight wasn't cancelled or diverted, it must have departure and arrival times.
# This drops rows that have missing operational data due to logging errors.
if 'Cancelled' in df.columns and 'Diverted' in df.columns:
    critical_cols = ['DepTime', 'ArrTime', 'ActualElapsedTime']
    existing_crit = [col for col in critical_cols if col in df.columns]
    
    # Keep row if it was cancelled/diverted OR if it has valid flight times
    is_cancelled_or_diverted = (df['Cancelled'] == 1) | (df['Diverted'] == 1)
    has_valid_times = df[existing_crit].notnull().all(axis=1)
    
    df = df[is_cancelled_or_diverted | has_valid_times]

# 8. Optimize Data Types (Saves massive memory)
# Convert binary flags to boolean
bool_cols = ['Cancelled', 'Diverted']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(bool)

# Downcast floats and ints where possible
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')
for col in df.select_dtypes(include=['int64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')

# Convert low-cardinality strings (like Carrier, Origin, Dest) to categories
cat_cols = ['Marketing_Airline_Network', 'Operating_Airline', 'Origin', 'Dest', 'Tail_Number']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# 9. Parse Dates (if applicable)
if 'FlightDate' in df.columns:
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])

# 10. Final Summary
print("\n--- Cleaning Complete ---")
print(f"Final Shape: {df.shape}")
print("\nRemaining Missing Values per Column:")
print(df.isnull().sum())

# Optional: Save the cleaned dataset
# df.to_csv("Flights_2022_Cleaned.csv", index=False)

# Calculate the threshold (e.g., keep columns with at least 10% non-null values)
threshold = 0.10 * len(df)
df.dropna(axis=1, thresh=threshold, inplace=True)

print(f"Shape after dropping columns with >90% missing values: {df.shape}")