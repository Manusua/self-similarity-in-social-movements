import csv
import json
import os
from collections import defaultdict

def create_hashtag_dict():
    # Define file paths
    base_dir = "datasets/data_hashtags_agosto_2023"
    files = [
        "hashtags_noaltarifazo.csv",
        "hashtags_ruidazonacional.csv"
    ]
    
    # Use defaultdict(set) to automatically handle duplicates and avoid checking existence
    hashtag_data = defaultdict(set)
    total_rows = 0

    for f in files:
        path = os.path.join(base_dir, f)
        if os.path.exists(path):
            print(f"Reading {path}...")
            try:
                with open(path, mode='r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        total_rows += 1
                        if total_rows % 1000000 == 0:
                            print(f"Processed {total_rows} rows...")
                        
                        # Extract id and hashtag
                        tweet_id = row.get('id')
                        hashtag = row.get('hashtag')
                        
                        if tweet_id and hashtag:
                            hashtag_data[tweet_id].add(hashtag)
            except Exception as e:
                print(f"Error reading {path}: {e}")
        else:
            print(f"Warning: {path} not found.")

    if hashtag_data:
        output_file = "datasets/dicts_hashtags/dict_hashtags_nat.json"
        print(f"Converting sets to lists and writing to {output_file}...")
        
        # Convert sets to lists for JSON serialization
        # This step creates a new dict in memory, effectively doubling memory usage briefly
        # To optimize, we can stream the JSON writing manually or just rely on swap
        final_data = {k: list(v) for k, v in hashtag_data.items()}
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
            
        print(f"Done. Processed {total_rows} rows into {len(final_data)} unique tweet IDs.")
    else:
        print("No data processed.")

if __name__ == "__main__":
    create_hashtag_dict()
