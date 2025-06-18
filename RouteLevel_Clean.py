import pandas as pd
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import re

def is_valid_icao(code):
    """
    Validate ICAO code format and exclude known invalid codes.

    Parameters:
        code: str, ICAO code to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(code, str):
        return False
    # ICAO codes: 4 letters, typically starting with region-specific letter
    # Exclude 'ZZZZ' and codes not starting with A-Z
    return bool(re.match(r'^[A-Z]{4}$', code)) and code != 'ZZZZ'

def clean_air_tsv(file_path, drop_missing=True):
    """
    Clean Eurostat aviation TSV data with enhanced ICAO validation.

    Parameters:
        file_path: str or Path
        drop_missing: bool, whether to drop rows with missing values

    Returns:
        pd.DataFrame: Cleaned long-format DataFrame with validated airport codes
    """
    try:
        df_raw = pd.read_csv(file_path, sep='\t', encoding='utf-8', engine='python', on_bad_lines='skip')

        # Validate first column
        if df_raw.empty or df_raw.columns[0] not in df_raw.columns:
            print(f"[ERROR] {file_path} has invalid or empty first column.")
            return pd.DataFrame()

        # Split composite fields in first column
        df_raw[['freq', 'unit', 'tra_meas', 'route']] = df_raw.iloc[:, 0].str.split(',', expand=True)
        df_raw.drop(columns=[df_raw.columns[0]], inplace=True)

        # Convert to long format
        df_long = df_raw.melt(
            id_vars=['freq', 'unit', 'tra_meas', 'route'],
            var_name='time_period',
            value_name='value'
        )

        # Replace missing symbols and convert to numeric
        df_long['value'] = df_long['value'].replace(':', pd.NA)
        df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')

        if drop_missing:
            initial_rows = df_long.shape[0]
            df_long.dropna(subset=['value'], inplace=True)
            print(f"[INFO] Dropped {initial_rows - df_long.shape[0]} rows with missing values in {file_path}")

        # Validate route column
        print(f"[INFO] Route value counts in {file_path} (top 5): {df_long['route'].value_counts().head(5)}")
        print(f"[INFO] Empty or invalid routes in {file_path}: {df_long['route'].isna().sum()}")

        # Extract airport codes
        df_long[['origin_airport', 'destination_airport']] = df_long['route'].str.extract(
            r'([A-Z]{2}_[A-Z]{4})_([A-Z]{2}_[A-Z]{4})'
        )

        # Extract ICAO codes
        df_long['origin_icao'] = df_long['origin_airport'].str.extract(r'([A-Z]{4})$')
        df_long['dest_icao'] = df_long['destination_airport'].str.extract(r'([A-Z]{4})$')

        # Validate ICAO codes
        df_long['valid_origin'] = df_long['origin_icao'].apply(is_valid_icao)
        df_long['valid_dest'] = df_long['dest_icao'].apply(is_valid_icao)

        # Log invalid ICAO codes
        invalid_rows = df_long[~df_long['valid_origin'] | ~df_long['valid_dest']]
        if not invalid_rows.empty:
            invalid_count = len(invalid_rows)
            print(f"[WARNING] Found {invalid_count} rows with invalid ICAO codes in {file_path}")
            # Save unique invalid routes
            invalid_routes = invalid_rows[['route', 'origin_icao', 'dest_icao']].drop_duplicates()
            invalid_routes.to_csv(f"output/invalid_routes_{Path(file_path).stem}.csv", index=False)
            print(f"[INFO] Saved {len(invalid_routes)} unique invalid routes to output/invalid_routes_{Path(file_path).stem}.csv")
            # Print sample of invalid codes
            sample_invalid = invalid_rows['dest_icao'].value_counts().head(5)
            print(f"[WARNING] Top 5 invalid destination ICAO codes: {sample_invalid.to_dict()}")

        # Drop rows with invalid ICAO codes
        initial_rows = df_long.shape[0]
        df_long = df_long[df_long['valid_origin'] & df_long['valid_dest']]
        print(f"[INFO] Dropped {initial_rows - df_long.shape[0]} rows with invalid ICAO codes in {file_path}")

        # Drop temporary validation columns
        df_long.drop(columns=['valid_origin', 'valid_dest', 'route'], inplace=True)

        return df_long

    except Exception as e:
        print(f"[ERROR] Failed to process file {file_path}: {e}")
        return pd.DataFrame()

def load_airport_coordinates(file_path):
    """
    Load airport coordinates from a dat file with validation.

    Parameters:
        file_path: str or Path

    Returns:
        pd.DataFrame: DataFrame with validated ICAO, Latitude, Longitude, Country
    """
    try:
        airports_df = pd.read_csv(file_path, header=None)
        airports_df.columns = [
            'Airport_ID', 'Name', 'City', 'Country', 'IATA', 'ICAO',
            'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST',
            'Tz_database_time_zone', 'Type', 'Source'
        ]
        # Validate ICAO codes
        airports_df['valid_icao'] = airports_df['ICAO'].apply(is_valid_icao)
        df = airports_df[airports_df['valid_icao']][['ICAO', 'Latitude', 'Longitude', 'Country']].dropna()
        print(f"[INFO] Loaded {len(df)} valid airport coordinates from {file_path}")
        if len(df) < len(airports_df):
            print(f"[WARNING] Dropped {len(airports_df) - len(df)} invalid or missing ICAO codes in {file_path}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return pd.DataFrame()

def merge_with_coordinates(df_all, airports_geo):
    """
    Merge route data with airport coordinates.

    Parameters:
        df_all: pd.DataFrame, cleaned route data
        airports_geo: pd.DataFrame, airport coordinates

    Returns:
        pd.DataFrame: Merged DataFrame with coordinates
    """
    df = df_all.copy().astype({'origin_icao': 'string', 'dest_icao': 'string'})

    # Merge origin airport info
    df = df.merge(
        airports_geo.rename(columns={
            'ICAO': 'origin_icao',
            'Latitude': 'origin_lat',
            'Longitude': 'origin_lon',
            'Country': 'origin_country'
        }),
        on='origin_icao',
        how='left'
    )

    # Merge destination airport info
    df = df.merge(
        airports_geo.rename(columns={
            'ICAO': 'dest_icao',
            'Latitude': 'dest_lat',
            'Longitude': 'dest_lon',
            'Country': 'dest_country'
        }),
        on='dest_icao',
        how='left'
    )

    # Check for missing coordinates
    missing_coords = df[df['origin_lat'].isna() | df['dest_lat'].isna()]
    if not missing_coords.empty:
        print(f"[WARNING] {len(missing_coords)} rows missing coordinates after merge.")
        # Save unique missing coordinates
        unique_missing = missing_coords[['origin_icao', 'dest_icao']].drop_duplicates()
        unique_missing.to_csv("output/missing_coordinates.csv", index=False)
        print(f"[INFO] Saved {len(unique_missing)} unique missing coordinates to output/missing_coordinates.csv")
        # Print summary of missing codes
        missing_dest_counts = missing_coords['dest_icao'].value_counts().head(5)
        print(f"[WARNING] Top 5 destination ICAO codes with missing coordinates: {missing_dest_counts.to_dict()}")

    return df

def save_by_decade(df, output_dir=".", file_format="parquet"):
    """
    Save route data by decade.

    Parameters:
        df: pd.DataFrame, enriched route data
        output_dir: str or Path, output directory
        file_format: str, output file format (parquet, csv, csv.gz)

    Returns:
        None
    """
    df = df.copy()
    df['year'] = df['time_period'].astype(str).str.extract(r'^(\d{4})')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    df['decade'] = (df['year'] // 10) * 10
    df['decade_label'] = df['decade'].astype(str) + 's'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label in df['decade_label'].unique():
        df_decade = df[df['decade_label'] == label]
        file_path = output_dir / f"routes_{label}.{file_format}"

        if file_format == 'parquet':
            df_decade.to_parquet(file_path, index=False)
        elif file_format == 'csv':
            df_decade.to_csv(file_path, index=False)
        elif file_format == 'csv.gz':
            df_decade.to_csv(file_path, index=False, compression='gzip')
        else:
            print(f"[ERROR] Unsupported format: {file_format}")
            continue

        print(f"[✔] Saved {file_path} ({len(df_decade)} rows)")

# ========= MAIN EXECUTION ==========

if __name__ == "__main__":
    tsv_dir = Path("/Users/gaoshitan/Desktop/Python/Pycharm/Master_Thesis/Master_Thesis/data")
    tsv_files = list(tsv_dir.glob("estat_avia_par_*.tsv"))

    # Check TSV files for consistency
    for file in tsv_files:
        try:
            df_sample = pd.read_csv(file, sep='\t', nrows=10)
            print(f"[INFO] Sample routes in {file}: {df_sample.iloc[:, 0].head(10).tolist()}")
        except Exception as e:
            print(f"[ERROR] Failed to read sample from {file}: {e}")

    # Process TSV files in parallel
    with ProcessPoolExecutor() as executor:
        # Map clean_air_tsv to each file and collect non-empty DataFrames
        results = executor.map(clean_air_tsv, tsv_files)
        # Filter out empty DataFrames and convert to list
        df_list = [df for df in results if not df.empty]

    if df_list:
        # Concatenate all non-empty DataFrames
        df_all = pd.concat(df_list, ignore_index=True)
        print(f"[INFO] Total rows after cleaning: {len(df_all)}")
    else:
        print("[ERROR] No valid DataFrames to concatenate.")
        df_all = pd.DataFrame()

    # Load airport coordinates and merge
    airports_geo = load_airport_coordinates(tsv_dir / "airports.dat")
    df_enriched = merge_with_coordinates(df_all, airports_geo)

    # Save by decade to compressed parquet files
    save_by_decade(df_enriched, output_dir="output", file_format="parquet")