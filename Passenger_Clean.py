import pandas as pd
from pathlib import Path


def clean_nuts2_passenger_data(file_path: str, drop_missing: bool = True) -> pd.DataFrame:
    """
    Clean Eurostat NUTS2-level passenger data (e.g., tran_r_avpa_nm.tsv)

    Parameters:
        file_path (str): Path to the raw TSV file
        drop_missing (bool): Whether to drop rows with missing values

    Returns:
        pd.DataFrame: Standardized table with ['geo', 'year', 'value']
    """
    try:
        # Load raw TSV file
        df = pd.read_csv(
            file_path, sep='\t', encoding='utf-8', engine='python', on_bad_lines='skip'
        )

        # Validate and split composite identifier column
        if not df.columns[0].count(',') >= 3:  # Expect freq,tra_meas,unit,geo
            raise ValueError(f"Invalid first column format in {file_path}: {df.columns[0]}")
        split_columns = df.columns[0].split(',')
        df[split_columns] = df[df.columns[0]].str.split(',', expand=True)
        df.drop(columns=[df.columns[0]], inplace=True)

        # Transform from wide to long format
        df_long = df.melt(
            id_vars=['freq', 'tra_meas', 'unit', 'geo\\TIME_PERIOD'],
            var_name='year',
            value_name='value'
        )

        # Rename fields
        df_long.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

        # Filter years early (2000–2023)
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        df_long['year'] = df_long['year'].str.strip()
        invalid_year_rows = df_long[~df_long['year'].str.match(r'^20\d{2}$', na=False)]
        if not invalid_year_rows.empty:
            invalid_year_rows[['geo', 'year']].to_csv(output_dir / "invalid_nuts2_years.csv", index=False)
            print(f"[INFO] Saved {len(invalid_year_rows)} invalid year rows to output/invalid_nuts2_years.csv")
        df_long = df_long[df_long['year'].str.match(r'^20\d{2}$', na=False)]

        # Clean value column
        df_long['value'] = df_long['value'].replace(':', pd.NA)
        df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')

        # Save missing value diagnostics
        if drop_missing:
            initial_rows = df_long.shape[0]
            missing_rows = df_long[df_long['value'].isna()][['geo', 'year']]
            if not missing_rows.empty:
                missing_rows.to_csv(output_dir / "missing_nuts2_values.csv", index=False)
                print(
                    f"[INFO] Saved {len(missing_rows)} missing value rows to "
                    f"output/missing_nuts2_values.csv"
                )
            df_long.dropna(subset=['value'], inplace=True)
            dropped_rows = initial_rows - df_long.shape[0]
            print(
                f"[INFO] Dropped {dropped_rows} rows with missing 'value' "
                f"({dropped_rows / initial_rows:.2%})"
            )

        # Clean and validate year
        df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce').astype('Int64')
        df_long = df_long[df_long['year'].notna() & (df_long['year'].between(2000, 2023))]

        # Keep only NUTS2-level regions: 4- or 5-character geo codes
        df_long['geo'] = df_long['geo'].astype(str)
        invalid_geo = df_long[~df_long['geo'].str.match(r'^[A-Z]{2}\d{2,3}$', na=False)]['geo'].unique()
        if invalid_geo.size > 0:
            print(f"[WARNING] Invalid geo codes found: {invalid_geo}")
            invalid_geo_rows = df_long[~df_long['geo'].str.match(r'^[A-Z]{2}\d{2,3}$', na=False)][['geo', 'year']]
            invalid_geo_rows.to_csv(output_dir / "invalid_nuts2_geo.csv", index=False)
            print(f"[INFO] Saved {len(invalid_geo_rows)} invalid geo rows to output/invalid_nuts2_geo.csv")
        df_long = df_long[df_long['geo'].str.match(r'^[A-Z]{2}\d{2,3}$', na=False)]

        # Report valid NUTS2 regions
        valid_geo_count = df_long['geo'].nunique()
        print(f"[INFO] Retained {valid_geo_count} valid NUTS2 regions")

        # Validate output
        if df_long.empty:
            print(f"[WARNING] No valid NUTS2 data after cleaning {file_path}")
            return pd.DataFrame()

        return df_long[['geo', 'year', 'value']]

    except FileNotFoundError:
        print(f"[❌] File not found: {file_path}")
        return pd.DataFrame()
    except pd.errors.ParserError:
        print(f"[❌] Parsing failed for {file_path}. Check file format.")
        return pd.DataFrame()
    except Exception as e:
        print(f"[❌] Unexpected error: {e}")
        return pd.DataFrame()


def analyze_missing_nuts2_values(missing_file: str):
    """
    Analyze missing values in NUTS2 passenger data.

    Parameters:
        missing_file (str): Path to missing_nuts2_values.csv
    """
    try:
        missing_df = pd.read_csv(missing_file)
        total_rows = len(missing_df)
        print(f"[INFO] Loaded {total_rows} rows from {missing_file}")
        # Summarize by geo
        geo_counts = missing_df['geo'].value_counts()
        geo_missing_ratio = geo_counts / geo_counts.sum()
        print(f"[INFO] Top 5 regions with missing values (count, ratio):\n"
              f"{pd.DataFrame({'Count': geo_counts.head(5), 'Ratio': geo_missing_ratio.head(5)})}")
        # Summarize by year
        year_counts = missing_df['year'].value_counts()
        year_missing_ratio = year_counts / year_counts.sum()
        print(f"[INFO] Top 5 years with missing values (count, ratio):\n"
              f"{pd.DataFrame({'Count': year_counts.head(5), 'Ratio': year_missing_ratio.head(5)})}")
        # Check regions with complete missing data
        regions_all_missing = geo_counts[geo_counts >= 24].index  # Assume 24 years
        if not regions_all_missing.empty:
            print(f"[WARNING] {len(regions_all_missing)} regions have no data for all years: {regions_all_missing}")
    except Exception as e:
        print(f"[❌] Failed to analyze {missing_file}: {e}")


# Example usage
if __name__ == "__main__":
    file_path = "data/estat_tran_r_avpa_nm.tsv"
    df_nuts2 = clean_nuts2_passenger_data(file_path)
    if not df_nuts2.empty:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        df_nuts2.to_csv(output_dir / "nuts2_passenger_demand.csv", index=False)
        print(f"[✔] Saved cleaned data to output/nuts2_passenger_demand.csv")
        # Analyze missing values
        missing_file = output_dir / "missing_nuts2_values.csv"
        if missing_file.exists():
            analyze_missing_nuts2_values(missing_file)
    else:
        print("[❌] No data to save.")