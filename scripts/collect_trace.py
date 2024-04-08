from pathlib import Path
import os
import pandas as pd
import numpy as np

# Example thresholds
HIGH_ACTIVITY_THRESHOLD = 20  # This is an example value
LOW_ACTIVITY_THRESHOLD = 2    # This is an example value

target_types = ['Sporadic', 'Bursty', 'Periodic']

# Example function to classify a row
def classify_trace(row):
    # Convert row to a list of counts, excluding non-numeric columns
    counts = row[4:].values
    # Define high and low activity based on thresholds
    high_activity_counts = counts > HIGH_ACTIVITY_THRESHOLD
    low_activity_counts = counts <= LOW_ACTIVITY_THRESHOLD
    
    non_zero_counts = np.count_nonzero(counts)
    total_counts = len(counts)
    percent_non_zero = non_zero_counts / total_counts * 100
    
    # Sporadic: 30% to 50% non-zero counts, but not consistently high
    if 30 <= percent_non_zero <= 50 and not high_activity_counts.any() and np.any(counts > LOW_ACTIVITY_THRESHOLD):
        return 'Sporadic'
    
    # # Bursty: Use a sliding window to find continuous high or low activity periods
    # continuous_high = np.convolve(high_activity_counts, np.ones(60, dtype=int), 'same') > 30  # 1 minute of high activity
    # continuous_low = np.convolve(low_activity_counts, np.ones(360, dtype=int), 'same') > 180  # 6 minutes of low activity
    # if continuous_high.sum() / total_counts >= 0.3 and continuous_low.sum() / total_counts >= 0.4:
    #     return 'Bursty'
    
    # Bursty: Identified by significant portions of both high and low activity
    # We found a periodic from it...
    if np.mean(high_activity_counts) >= 0.2 and np.mean(low_activity_counts) >= 0.4 and 100 <= counts.max() <= 500:
        return 'Bursty'
    
    # # Periodic: Alternating between high and low, this is simplistic and might need fine-tuning
    # changes = np.count_nonzero(np.diff(counts > LOW_ACTIVITY_THRESHOLD))  # Count changes from high to low
    # if changes > 24:  # Assuming more than 24 changes in 24 hours indicates periodic behavior
    #     return 'Periodic'
    
    # Periodic: Counts repeatedly transition between above high and below low thresholds
    # Find transitions from high to low and low to high
    transitions_high_to_low = ((high_activity_counts[:-1] == True) & (low_activity_counts[1:] == True)).sum()
    transitions_low_to_high = ((low_activity_counts[:-1] == True) & (high_activity_counts[1:] == True)).sum()

    # Define "many times" as a specific number of transitions to consider as periodic
    if transitions_high_to_low > 24 and transitions_low_to_high > 24:  # Example threshold
        return 'Periodic'
    
    return 'Unknown'

def main():
    # Load the data
    file_path = os.path.join(str(Path.home()), "download/trace/invocations_per_function_md.anon.d13.csv")
    df = pd.read_csv(file_path)
    # Find rows where the minimum count across all columns is greater than 60
    always_hot_mask = df.iloc[:, 4:].min(axis=1) > 0
    # Filter out "always hot" traces
    df = df[~always_hot_mask]
    # Apply classification to each row
    df['Classification'] = df.apply(classify_trace, axis=1)
    for x in target_types:
        # Filter unknown rows
        df_filtered = df[df['Classification'] == x]
        # Print the classification results
        print(df_filtered[['HashFunction', 'Classification']])
    
if __name__ == "__main__":
    main()