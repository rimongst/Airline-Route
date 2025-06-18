import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS, RandomEffects
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from RouteLevel_Consistency import merge_parquet_to_csv
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Set random seed for reproducibility
np.random.seed(42)

# --- Data Processing Functions ---
def fix_eurostat_table(file_path, value_name):
    """Fix Eurostat TSV file by extracting year and converting values."""
    if not Path(file_path).is_file():
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    df_raw = pd.read_csv(file_path, sep='\t', encoding='utf-8', engine='python')
    df = df_raw.copy()
    df[['freq', 'unit', 'region_code']] = df[df.columns[0]].str.split(',', expand=True)
    df.drop(columns=[df.columns[0]], inplace=True)

    df_long = df.melt(id_vars='region_code', var_name='year', value_name=value_name)
    df_long['year'] = df_long['year'].astype(str).str.extract(r'(\d{4})')
    df_long = df_long.dropna(subset=['year'])
    df_long['year'] = df_long['year'].astype(int)

    df_long[value_name] = df_long[value_name].replace([':', '-', ' '], np.nan)
    df_long[value_name] = pd.to_numeric(df_long[value_name].str.replace(r'[^\d\.]', '', regex=True), errors='coerce')
    df_long['region_code'] = df_long['region_code'].astype(str).str.strip()
    return df_long

def load_airport_mapping(path):
    """Load airport to NUTS2 mapping with error handling."""
    if not Path(path).is_file():
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    return df.rename(columns={'airport_code': 'origin_icao', 'nuts2_region': 'origin_nuts2'}).assign(
        origin_nuts2=lambda x: x['origin_nuts2'].astype(str).str.strip()
    )

def build_route_region_data(parquet_paths, airport_map_path):
    """Build region-level route count data."""
    df = merge_parquet_to_csv(parquet_paths)
    airport_map = load_airport_mapping(airport_map_path)
    df = df.merge(airport_map, on='origin_icao', how='left')
    df['route_key'] = df['origin_icao'] + "_" + df['dest_icao'] + "_" + df['year'].astype(str)
    df_region = df.groupby(['origin_nuts2', 'year'])['route_key'].nunique().reset_index()
    return df_region.rename(columns={
        'origin_nuts2': 'region_code',
        'route_key': 'route_count'
    }).assign(region_code=lambda x: x['region_code'].astype(str).str.strip())

def load_passenger(path):
    """Load passenger demand data with error handling."""
    if not Path(path).is_file():
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    return df.rename(columns={'geo': 'region_code', 'value': 'passenger'}).assign(
        year=lambda x: x['year'].astype(int),
        region_code=lambda x: x['region_code'].astype(str).str.strip()
    )

def construct_mismatch(df_passenger, df_route):
    """Construct mismatch between passenger and route growth."""
    df = pd.merge(df_passenger, df_route, on=['region_code', 'year'], how='inner')
    if df.empty:
        logging.error("No data after merging passenger and route data")
        return df
    df = df.sort_values(['region_code', 'year'])
    df['passenger_growth'] = df.groupby('region_code')['passenger'].pct_change()
    df['route_growth'] = df.groupby('region_code')['route_count'].pct_change()
    df['mismatch'] = np.where(
        (df['passenger_growth'] > 0) & (df['route_growth'] < 0), 1,
        np.where((df['passenger_growth'] < 0) & (df['route_growth'] > 0), 1, 0)
    )
    return df

def enrich_with_controls(df, pop_path, gdp_path, hub_regions):
    """Enrich dataset with control variables."""
    df['region_code'] = df['region_code'].astype(str).str.strip()
    df_pop = fix_eurostat_table(pop_path, 'pop_density')
    df_gdp = fix_eurostat_table(gdp_path, 'gdp_per_capita')

    df = df.merge(df_pop, on=['region_code', 'year'], how='left')
    df = df.merge(df_gdp, on=['region_code', 'year'], how='left')
    df['is_hub_region'] = df['region_code'].isin(hub_regions).astype(int)
    df['interaction'] = df['passenger_growth'] * df['is_hub_region']
    return df.dropna(subset=['passenger_growth', 'route_growth', 'pop_density', 'gdp_per_capita'])

# --- Modeling Functions ---
def check_multicollinearity(df, features):
    """Check multicollinearity using VIF."""
    logging.info("\n📊 Checking VIF for multicollinearity:")
    df = df.dropna(subset=features).copy()
    if df.empty:
        logging.warning("❗️ No valid rows after dropping NA in VIF check. Skipping.")
        return None
    X = sm.add_constant(df[features])
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    logging.info(vif.to_string())
    return vif

def run_enhanced_model(df, save_path="figures/period_regression_coefficients.png"):
    """Run PanelOLS model with entity and time effects for different time periods."""
    # Define time periods
    periods = {
        "Pre-Pandemic (<2020)": df[df['year'] < 2020],
        "Pandemic (2020-2022)": df[(df['year'] >= 2020) & (df['year'] <= 2022)],
        "Post-Pandemic (>2022)": df[df['year'] > 2022]
    }

    results_dict = {}
    coef_data = []

    for period_name, df_period in periods.items():
        logging.info(f"\n📊 Running regression for {period_name} ({df_period.shape[0]} rows)")
        df_model = df_period.dropna(subset=['mismatch', 'passenger_growth', 'gdp_per_capita', 'pop_density']).copy()

        if df_model.empty:
            logging.warning(f"❌ No valid data for {period_name}. Skipping.")
            continue

        # Check multicollinearity
        check_multicollinearity(df_model, ['interaction', 'gdp_per_capita', 'pop_density'])

        # Run PanelOLS
        df_model = df_model.set_index(['region_code', 'year'])
        formula = 'mismatch ~ interaction + gdp_per_capita + pop_density + EntityEffects + TimeEffects'
        try:
            model = PanelOLS.from_formula(formula, data=df_model, check_rank=False)
            results = model.fit()
            logging.info(f"{period_name} Results:\n{results.summary}")
            results_dict[period_name] = results

            # Collect coefficients for visualization
            for var in ['interaction', 'gdp_per_capita', 'pop_density']:
                coef = results.params.get(var, np.nan)
                se = results.std_errors.get(var, np.nan)
                coef_data.append({
                    'Period': period_name,
                    'Variable': var,
                    'Coefficient': coef,
                    'Std_Error': se
                })
        except Exception as e:
            logging.error(f"Error in {period_name} regression: {e}")
            continue

    # Visualize coefficients across periods
    if coef_data:
        coef_df = pd.DataFrame(coef_data)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Variable', y='Coefficient', hue='Period', data=coef_df)
        plt.errorbar(
            x=[i + offset for i, offset in zip(range(len(coef_df)), [-0.2, 0, 0.2] * (len(coef_df) // 3))],
            y=coef_df['Coefficient'],
            yerr=coef_df['Std_Error'],
            fmt='none', c='black', capsize=5
        )
        plt.title('Regression Coefficients by Time Period')
        plt.ylabel('Coefficient Value')
        plt.xlabel('Variable')
        plt.legend(title='Period')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logging.info(f"Saved coefficient comparison plot to {save_path}")
        plt.close()

    return results_dict

def run_random_effects_model(df):
    """Run Random Effects model for robustness check."""
    df_model = df.dropna(subset=['mismatch', 'passenger_growth', 'gdp_per_capita', 'pop_density']).copy()
    if df_model.empty:
        logging.warning("❌ ERROR: No valid data for Random Effects model. Exiting.")
        return None

    df_model = df_model.set_index(['region_code', 'year'])
    exog_vars = ['interaction', 'gdp_per_capita', 'pop_density']
    exog = sm.add_constant(df_model[exog_vars])

    logging.info(f"Exog shape: {exog.shape}")
    logging.info(f"Exog columns: {exog.columns}")
    logging.info(f"Exog rank: {np.linalg.matrix_rank(exog)}")

    model = RandomEffects(df_model['mismatch'], exog, check_rank=False)
    results = model.fit()
    logging.info("[INFO] Random Effects Model Summary:")
    logging.info(results.summary)
    return results

# --- Visualization Functions ---
def plot_mismatch_by_hub(df, save_path="figures/mismatch_by_hub.png"):
    """Plot mismatch rate for hub vs non-hub regions."""
    mismatch_rate = df.groupby('is_hub_region')['mismatch'].mean().reset_index()
    mismatch_rate['Region Type'] = mismatch_rate['is_hub_region'].map({0: 'Non-Hub', 1: 'Hub'})

    plt.figure(figsize=(6, 4))
    sns.barplot(x='Region Type', y='mismatch', data=mismatch_rate)
    plt.title('Average Mismatch Rate: Hub vs Non-Hub')
    plt.ylabel('Mismatch Rate')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logging.info(f"[INFO] Saved mismatch bar chart to {save_path}")
    plt.close()

def plot_growth_scatter(df, save_path="figures/growth_vs_mismatch.png"):
    """Scatter plot: passenger growth vs mismatch, colored by hub type."""
    df_plot = df[df['passenger_growth'] < 20].copy()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='passenger_growth', y='mismatch', hue='is_hub_region', data=df_plot, alpha=0.6)
    plt.title('Passenger Growth vs Mismatch (Zoomed)')
    plt.xlabel('Passenger Growth (<20)')
    plt.ylabel('Mismatch')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logging.info(f"[INFO] Saved scatter plot to {save_path}")
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    # Configuration
    passenger_path = "output/nuts2_passenger_demand.csv"
    mapping_path = "data/airport_to_nuts2_mapping.csv"
    parquet_paths = [
        "output/routes_2000s.parquet",
        "output/routes_2010s.parquet",
        "output/routes_2020s.parquet"
    ]
    hub_regions = ['FR10', 'DE30', 'NL32', 'ITC4', 'ES30', 'BE10', 'PL12', 'IE06', 'AT13', 'SE11']

    # Data processing
    df_passenger = load_passenger(passenger_path)
    df_route = build_route_region_data(parquet_paths, mapping_path)
    df_base = construct_mismatch(df_passenger, df_route)
    if df_base.empty:
        logging.error("No data after constructing mismatch. Check input files.")
        exit(1)

    df_enriched = enrich_with_controls(
        df_base,
        pop_path='data/estat_demo_r_d3dens.tsv',
        gdp_path='data/estat_nama_10r_3gdp.tsv',
        hub_regions=hub_regions
    )
    if df_enriched.empty:
        logging.error("No data after enriching controls. Check Eurostat files.")
        exit(1)

    # Save enriched data
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    df_enriched.to_csv(output_dir / "region_level_enriched.csv", index=False)

    # Run models
    run_enhanced_model(df_enriched, save_path="figures/period_regression_coefficients.png")
    run_random_effects_model(df_enriched)

    # Generate plots
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    plot_mismatch_by_hub(df_enriched, save_path=figures_dir / "mismatch_by_hub.png")
    plot_growth_scatter(df_enriched, save_path=figures_dir / "growth_vs_mismatch.png")