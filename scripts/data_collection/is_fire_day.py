import pandas as pd


import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np

# Load the CSV files
try:
    df_A = pd.read_csv('/Users/teodoravujovic/Desktop/code/firebird/processed_data.csv')
    df_B = pd.read_csv('/Users/teodoravujovic/Desktop/data/firebird/historical_wildfires.csv')[:25858]
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Make sure the CSV files are in the correct directory.")
    exit()

# Rename 'FIRE_START_DATE' to 'date' in df_B for consistent merging/grouping
# df_B = df_B.rename(columns={'FIRE_START_DATE': 'date'})
# Method 4: Formatting as a string (if you need a string representation)
# Be cautious with this method if you plan further date calculations, as it converts to string.
df_B['FIRE_START_DATE'] = pd.to_datetime(df_B['FIRE_START_DATE'])
df_B['date'] = df_B['FIRE_START_DATE'].dt.strftime('%Y-%m-%d')
print(df_B.date.unique())

# Define the radius for the proximity search (in degrees)
# As per your request, 0.305 degrees for both lat and lon difference.
RADIUS_DEGREE = 0.305

# Initialize the 'is_fire_day' column in df_A to 0
df_A['is_fire_day'] = 0

print("Starting proximity search and 'is_fire_day' column generation...")

# Group df_B by date to process day by day for efficiency
# This avoids building a BallTree for all ignition points at once, which would be less efficient
# if you're only interested in points on the same date.
df_B_grouped_by_date = df_B.groupby('date')

# Iterate through each unique date in df_A
for current_date in df_A['date'].unique():
    # Filter df_A for the current date
    df_A_on_date = df_A[df_A['date'] == current_date]
    print(current_date)

    # Get ignition points for the current date from df_B
    # Use .get_group() for efficiency with groupby objects
    try:
        df_B_on_date = df_B_grouped_by_date.get_group(current_date)
    except KeyError:
        # No ignitions on this date, so all 'is_fire_day' for this date will remain 0
        continue

    # Prepare coordinates for BallTree: [latitude, longitude]
    # BallTree works best with spherical coordinates if true distance is needed,
    # but for simple degree-based proximity, Cartesian is fine for lat/lon directly.
    # We'll use the direct lat/lon as 'features' for the BallTree here, essentially creating
    # a 2D index for quick rectangular queries.
    ignition_coords = df_B_on_date[['LATITUDE', 'LONGITUDE']].values
    query_coords = df_A_on_date[['latitude', 'longitude']].values

    # Build a BallTree for the ignition points on the current date
    # Using 'metric="chebyshev"' is suitable for "max difference" (square/rectangle radius)
    # in lat/lon space, effectively checking if |lat1-lat2| <= radius AND |lon1-lon2| <= radius.
    tree = BallTree(ignition_coords, metric='chebyshev')

    # Query the BallTree for points within the specified radius
    # The 'query_radius' method returns a list of arrays, where each array contains the indices
    # of neighbors within the radius for the corresponding query point.
    # The 'count_only=True' option makes it faster if you only need to know if there's *any* neighbor.
    ind = tree.query_radius(query_coords, r=RADIUS_DEGREE, count_only=False)

    # For each query point in df_A_on_date, if there's at least one ignition point within the radius,
    # mark it as a 'fire day'.
    fire_day_indices = df_A_on_date.index[np.array([len(i) > 0 for i in ind])]

    # Update the 'is_fire_day' column in the original df_A
    df_A.loc[fire_day_indices, 'is_fire_day'] = 1

print("\n'is_fire_day' column successfully added.")
print("\nDataFrame A with 'is_fire_day' column (first 100 rows):")
print(df_A.head(100))

# Optional: Display some statistics
print(f"\nTotal rows in df_A: {len(df_A)}")
print(f"Number of 'is_fire_day' marked as 1: {df_A['is_fire_day'].sum()}")
print(f"Number of 'is_fire_day' marked as 0: {len(df_A) - df_A['is_fire_day'].sum()}")

# Optional: Display some statistics
print(f"\nTotal rows in January 2006 of df_A: {len(df_A[:1258*31])}")
print(f"Number of 'is_fire_day' marked as 1: {df_A['is_fire_day'][:1258*31].sum()}")
print(f"Number of 'is_fire_day' marked as 0: {1258*31 - df_A['is_fire_day'][:1258*31].sum()}")

df_A.to_csv('./processed_data.csv')
print('All Done!')

# Example to check a specific entry
# You might want to pick a known ignition location/date from df_B and check nearby points in df_A
# For instance, if df_B has an ignition at (40.0, -105.0) on 2023-01-01
# You would expect a row in df_A for 2023-01-01 and lat/lon within 0.305 of (40.0, -105.0) to be 1.
# print(df_A[(df_A['date'] == pd.to_datetime('2023-01-01')) &
#            (df_A['latitude'] > 39.7) & (df_A['latitude'] < 40.3) &
#            (df_A['longitude'] > -105.3) & (df_A['longitude'] < -104.7)].head())
