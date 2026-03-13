"""
Fix CSV files to ensure all numeric columns are saved as proper floats
instead of scientific notation strings.
"""

import pandas as pd
import numpy as np

def fix_csv_dtypes(input_path, output_path):
    """
    Load CSV and ensure all numeric columns are properly typed
    """
    print(f"Processing {input_path}...")
    
    # Load CSV with converters to handle scientific notation
    df = pd.read_csv(input_path, float_precision='round_trip')
    
    print(f"Original dtypes:\n{df.dtypes}\n")
    
    # Explicitly convert numeric columns
    numeric_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                   'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                   'euribor3m', 'nr.employed']
    
    for col in df.columns:
        if col in numeric_cols or df[col].dtype in ['int64', 'float64']:
            # Force conversion to numeric, handling scientific notation strings
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Fixed dtypes:\n{df.dtypes}\n")
    
    # Save with proper formatting - NO scientific notation
    # Use high precision and disable scientific notation
    df.to_csv(output_path, index=False, float_format='%.10f')
    
    print(f"Saved to {output_path}\n")
    
    # Verify by reloading
    df_verify = pd.read_csv(output_path, float_precision='round_trip')
    print(f"Verification - dtypes after reload:\n{df_verify.dtypes}\n")
    
    # Check for any object columns that should be numeric
    object_cols = df_verify.select_dtypes(include=['object']).columns
    numeric_object_cols = [col for col in object_cols if col in numeric_cols]
    
    if len(numeric_object_cols) > 0:
        print(f"⚠️  Warning: These numeric columns are still object type: {list(numeric_object_cols)}")
        for col in numeric_object_cols:
            print(f"  {col}: {df_verify[col].head()}")
    else:
        print("✅ All numeric columns are properly typed!")
    
    return df

if __name__ == "__main__":
    # Fix train_df.csv
    fix_csv_dtypes(
        'data/processed/train_df.csv',
        'data/processed/train_df.csv'
    )
    
    # Fix test_df.csv
    fix_csv_dtypes(
        'data/processed/test_df.csv',
        'data/processed/test_df.csv'
    )
    
    # Fix processed_train_df.csv if it exists
    try:
        fix_csv_dtypes(
            'data/processed/processed_train_df.csv',
            'data/processed/processed_train_df.csv'
        )
    except FileNotFoundError:
        print("processed_train_df.csv not found, skipping...")
    
    print("\n" + "="*60)
    print("✅ ALL CSV FILES HAVE BEEN FIXED!")
    print("="*60)
    print("\nNow run:")
    print("  git add data/processed/*.csv")
    print("  git commit -m 'Fix CSV scientific notation'")
    print("  git push")
