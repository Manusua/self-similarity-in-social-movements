import pandas as pd
import json
import os

def count_hashtags():
    # Define file paths
    base_path = '/home/msuarez/Escritorio/self_similarity/main_branch/github/self-similarity-in-social-movements'
    files = [
        os.path.join(base_path, 'datasets/data_hashtags_agosto_2023/hashtags_ruidazonacional.csv'),
        os.path.join(base_path, 'datasets/data_hashtags_agosto_2023/hashtags_noaltarifazo.csv')
    ]
    output_dir = os.path.join(base_path, 'datasets/dicts_hashtags')
    output_file = os.path.join(output_dir, 'hashtags_nat_counts.json')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dfs = []
    print("Leyendo archivos CSV...")
    for file_path in files:
        if os.path.exists(file_path):
            try:
                # Read only the 'hashtag' column to save memory and time
                df = pd.read_csv(file_path, usecols=['hashtag'], dtype={'hashtag': str})
                dfs.append(df)
                print(f"Leído: {file_path} - {len(df)} filas")
            except Exception as e:
                print(f"Error leyendo {file_path}: {e}")
        else:
            print(f"Archivo no encontrado: {file_path}")

    if not dfs:
        print("No se pudieron leer datos.")
        return

    # Concatenate all dataframes
    print("Procesando y contando hashtags...")
    full_df = pd.concat(dfs, ignore_index=True)

    # Count occurrences
    hashtag_counts = full_df['hashtag'].value_counts()

    # Convert to dictionary (value_counts returns sorted descending by default)
    counts_dict = hashtag_counts.to_dict()

    # Write to JSON
    print(f"Guardando resultados en {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(counts_dict, f, ensure_ascii=False, indent=4)

    print("Proceso completado exitosamente.")

if __name__ == "__main__":
    count_hashtags()
