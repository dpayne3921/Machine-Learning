from datetime import datetime, timedelta
import pandas as pd

def check_sequential_dates(data):
    # Convert data to datetime objects if they are not already
    data = [pd.to_datetime(d) if not isinstance(d, datetime) else d for d in data]
    
    # Check if data is sorted
    if data != sorted(data):
        return False, "Data is not in sequential order."
    
    # Check for missing dates
    missing_dates = []
    current_date = data[0]
    end_date = data[-1]
    
    while current_date < end_date:
        current_date += timedelta(days=1)
        if current_date not in data:
            missing_dates.append(current_date)
    
    if missing_dates:
        return False, f"Data has missing dates: {missing_dates}"
    
    return True, "Data is in the correct form and has no missing dates."


def check_missing_values(df):
    try:
        # Print column names for debugging
        print("DataFrame columns:", df.columns)
        
        missing_counts = {}
        for column in df.columns:
            missing_counts[column] = df[column].isna().sum()
        
        return missing_counts
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

