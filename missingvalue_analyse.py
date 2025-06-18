import pandas as pd
from pathlib import Path

def analyze_missing_coordinates(missing_file, airports_file):
    """
    Analyze missing coordinates and check against airports data.

    Parameters:
        missing_file: str or Path, path to missing_coordinates.csv
        airports_file: str or Path, path to airports.dat
    """
    # Load missing coordinates
    missing_df = pd.read_csv(missing_file)
    print(f"[INFO] Loaded {len(missing_df)} rows from {missing_file}")

    # Summarize unique origin and destination codes
    unique_origins = missing_df['origin_icao'].unique()
    unique_dests = missing_df['dest_icao'].unique()
    print(f"[INFO] Unique origin ICAO codes: {len(unique_origins)}")
    print(f"[INFO] Unique destination ICAO codes: {len(unique_dests)}")
    print(f"[INFO] Top 5 destination ICAO codes:\n{missing_df['dest_icao'].value_counts().head(5)}")

    # Load airports data
    airports_df = pd.read_csv(airports_file, header=None)
    airports_df.columns = [
        'Airport_ID', 'Name', 'City', 'Country', 'IATA', 'ICAO',
        'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST',
        'Tz_database_time_zone', 'Type', 'Source'
    ]
    print(f"[INFO] Loaded {len(airports_df)} airports from {airports_file}")

    # Check if missing codes exist in airports data
    missing_origins_in_airports = airports_df[airports_df['ICAO'].isin(unique_origins)]
    missing_dests_in_airports = airports_df[airports_df['ICAO'].isin(unique_dests)]
    print(f"[INFO] Origins found in airports.dat: {len(missing_origins_in_airports)}/{len(unique_origins)}")
    print(f"[INFO] Destinations found in airports.dat: {len(missing_dests_in_airports)}/{len(unique_dests)}")

    # Save missing codes not in airports.dat
    not_in_airports = set(unique_dests) - set(airports_df['ICAO'])
    if not_in_airports:
        pd.Series(list(not_in_airports)).to_csv("output/missing_icao_not_in_airports.csv", index=False)
        print(f"[INFO] Saved {len(not_in_airports)} destination ICAO codes not in airports.dat to output/missing_icao_not_in_airports.csv")

    # Check specific codes (e.g., LYTI, ZZZZ)
    check_codes = ['LYTI', 'ZZZZ', 'SBRE', 'SBWR', 'SBWZ', 'SBXS', 'GVSC', 'LKAA', 'LTBB', 'KZMA']
    for code in check_codes:
        if code in airports_df['ICAO'].values:
            print(f"[INFO] {code} found in airports.dat")
        else:
            print(f"[WARNING] {code} not found in airports.dat")

if __name__ == "__main__":
    missing_file = "output/missing_coordinates.csv"
    airports_file = "/Users/gaoshitan/Desktop/Python/Pycharm/Master_Thesis/Master_Thesis/data/airports.dat"
    analyze_missing_coordinates(missing_file, airports_file)