import pandas as pd
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def prepare_enriched_route_data(route_path, airport_map_path, growth_path, output_dir="output"):
    """
    Merge route-level data with airport-region mapping and passenger growth.

    Parameters:
        route_path (str or Path): Path to route-level parquet file
        airport_map_path (str or Path): Path to airport-to-NUTS2 CSV
        growth_path (str or Path): Path to passenger growth CSV
        output_dir (str or Path): Directory to save enriched output

    Raises:
        FileNotFoundError: If input files are not found
        ValueError: If data validation fails
    """
    # Convert to Path objects and create output directory
    route_path = Path(route_path)
    airport_map_path = Path(airport_map_path)
    growth_path = Path(growth_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate input files
    for file_path in [route_path, airport_map_path, growth_path]:
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

    # 2. Load route-level consistency data
    logging.info(f"Loading route data from {route_path}")
    df = pd.read_parquet(route_path)

    # 3. Merge airport to NUTS2 mapping
    logging.info(f"Merging airport mapping from {airport_map_path}")
    airport_map = pd.read_csv(airport_map_path)
    if not all(col in airport_map.columns for col in ["airport_code", "nuts2_region"]):
        raise ValueError("airport_map_path must contain 'airport_code' and 'nuts2_region' columns")
    airport_map = airport_map.rename(columns={
        "airport_code": "origin_airport",
        "nuts2_region": "departure_region"
    })
    df = df.merge(airport_map, on="origin_airport", how="left")

    # 4. Merge NUTS2-level passenger growth
    logging.info(f"Merging passenger growth from {growth_path}")
    growth = pd.read_csv(growth_path)
    if not all(col in growth.columns for col in ["geo", "value", "year"]):
        raise ValueError("growth_path must contain 'geo', 'value', and 'year' columns")
    growth = growth.rename(columns={
        "geo": "departure_region",
        "value": "passenger_growth"
    })

    df = df.drop(columns=["passenger_growth"], errors="ignore")
    df = df.merge(growth, on=["departure_region", "year"], how="left")

    # 5. Validate and process passenger_growth
    if not pd.api.types.is_numeric_dtype(df['passenger_growth']):
        raise ValueError("passenger_growth must be numeric")
    missing = df['passenger_growth'].isna().sum()
    total = len(df)
    if missing > 0:
        logging.warning(f"Dropping {missing} rows ({missing/total*100:.1f}%) with missing passenger growth.")
        df = df.dropna(subset=['passenger_growth'])

    # 6. Save output
    output_path = output_dir / "enriched_route_level.parquet"
    logging.info(f"Saving enriched data to {output_path}")
    df.to_parquet(output_path, index=False)
    logging.info(f"✅ Enriched route-level data saved to {output_path} ({len(df)} rows)")

if __name__ == "__main__":
    try:
        prepare_enriched_route_data(
            route_path="output/route_consistency.parquet",
            airport_map_path="data/airport_to_nuts2_mapping.csv",
            growth_path="output/nuts2_passenger_demand.csv",
            output_dir="output"
        )
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise