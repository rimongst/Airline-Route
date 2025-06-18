import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from esda.moran import Moran
from esda.moran import Moran_Local
from libpysal.weights import Queen
from spreg import ML_Lag
from pathlib import Path
import logging
import json
import numpy as np
from matplotlib.legend_handler import HandlerPatch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Set random seed for reproducibility
np.random.seed(42)

# --- Data Processing Functions ---

def load_enriched_data(path='output/region_level_enriched.csv'):
    """Load enriched data from CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded and processed DataFrame.

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If required columns are missing.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    required_cols = ['region_code', 'mismatch', 'year', 'passenger_growth', 'pop_density', 'gdp_per_capita']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    df['region_code'] = df['region_code'].astype(str).str.strip().str.upper()
    logging.info(f"Loaded enriched data with {len(df)} rows, unique region_codes: {df['region_code'].nunique()}")
    return df

def load_nuts2_geojson(path='data/nuts2_boundaries.geojson'):
    """Load NUTS2 boundary data from GeoJSON file.

    Args:
        path (str): Path to the GeoJSON file.

    Returns:
        gpd.GeoDataFrame: Loaded and processed GeoDataFrame.

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If 'NUTS_ID' column is missing.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        gj = json.load(f)
    gdf = gpd.GeoDataFrame.from_features(gj["features"])
    if 'NUTS_ID' not in gdf.columns:
        raise ValueError("GeoJSON file must contain 'NUTS_ID' column")
    gdf = gdf.rename(columns={'NUTS_ID': 'region_code'})
    gdf['region_code'] = gdf['region_code'].astype(str).str.strip().str.upper()
    logging.info(f"Loaded GeoJSON with {len(gdf)} regions, unique region_codes: {gdf['region_code'].nunique()}")
    return gdf

# --- Visualization Functions ---

def plot_average_mismatch_map(df, gdf, xlim=(-25, 45), ylim=(33, 72), cmap='OrRd', save_path=None):
    """Plot average mismatch rate map for NUTS2 regions.

    Args:
        df (pd.DataFrame): Enriched data.
        gdf (gpd.GeoDataFrame): NUTS2 boundary data.
        xlim (tuple): Longitude range for clipping.
        ylim (tuple): Latitude range for clipping.
        cmap (str): Colormap for the plot.
        save_path (str, optional): Path to save the figure.
    """
    df_avg = df.groupby('region_code')['mismatch'].mean().reset_index()
    gdf = gdf.rename(columns={'NUTS_ID': 'region_code'}) if 'NUTS_ID' in gdf.columns else gdf
    gdf['region_code'] = gdf['region_code'].astype(str).str.strip().str.upper()
    gdf_merged = gdf.merge(df_avg, on='region_code', how='left')

    # Clip to Europe mainland
    gdf_merged = gdf_merged.set_geometry(gdf_merged.geometry.centroid)
    gdf_merged = gdf_merged.cx[xlim[0]:xlim[1], ylim[0]:ylim[1]]
    valid_regions = gdf_merged['region_code'].tolist()
    gdf_merged = gdf[gdf['region_code'].isin(valid_regions)].merge(df_avg, on='region_code', how='left')

    fig, ax = plt.subplots(figsize=(13, 7))
    gdf_merged.plot(
        column='mismatch',
        cmap=cmap,
        linewidth=0.2,
        edgecolor='gray',
        legend=True,
        legend_kwds={'label': "Average Mismatch Rate", 'shrink': 0.6, 'orientation': 'vertical'},
        ax=ax,
        missing_kwds={'color': 'lightgrey', 'label': 'No Data'}
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("Average Mismatch Rate by NUTS2 Region (Europe Mainland Only)", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved mismatch map to {save_path}")
    plt.close()

# --- Spatial Autocorrelation Analysis ---

def compute_morans_I(df, gdf):
    """Compute Moran's I global spatial autocorrelation.

    Args:
        df (pd.DataFrame): Enriched data.
        gdf (gpd.GeoDataFrame): NUTS2 boundary data.

    Returns:
        Moran: Moran object with results.
    """
    df_avg = df.groupby('region_code')['mismatch'].mean().reset_index()
    gdf_merged = gdf.merge(df_avg, on='region_code', how='left')
    gdf_merged = gdf_merged.dropna(subset=['mismatch'])

    w = Queen.from_dataframe(gdf_merged, use_index=True)
    w.transform = 'r'

    moran = Moran(gdf_merged['mismatch'], w)

    logging.info("\n📊 Moran’s I Global spatial autocorrelation test results：")
    logging.info(f"Moran's I: {moran.I:.4f}")
    logging.info(f"p-value: {moran.p_sim:.4f} (based on {moran.permutations} permutations)")
    logging.info(f"Expected I (E[I]): {moran.EI:.4f}")

    return moran

def run_lisa_analysis(df, gdf):
    """Run LISA analysis for local spatial autocorrelation.

    Args:
        df (pd.DataFrame): Enriched data.
        gdf (gpd.GeoDataFrame): NUTS2 boundary data.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with LISA results.
    """
    logging.info("Starting LISA analysis...")
    df_avg = df.groupby('region_code')['mismatch'].mean().reset_index()
    gdf['region_code'] = gdf['region_code'].astype(str).str.strip().str.upper()
    gdf = gdf.merge(df_avg, on='region_code', how='left')
    gdf = gdf.dropna(subset=['mismatch'])

    # Exclude islands
    island_ids = [7, 20, 48, 52, 60, 61, 68, 70, 91, 94, 134, 156, 157, 169, 178, 179, 216, 219, 221, 230, 325]
    gdf = gdf[~gdf.index.isin(island_ids)]

    # Create weights
    w = Queen.from_dataframe(gdf, use_index=True)
    w.transform = 'r'
    logging.info("Queen weights computed.")

    # Clip to Europe mainland
    gdf['centroid'] = gdf.geometry.centroid
    gdf = gdf.set_geometry('centroid')
    gdf_clipped = gdf.cx[-25:45, 33:72]
    gdf = gdf.set_geometry('geometry')
    gdf = gdf.loc[gdf_clipped.index].reset_index(drop=True)

    # Align weights with clipped data
    w_clipped = Queen.from_dataframe(gdf, use_index=True)
    w_clipped.transform = 'r'
    logging.info(f"Weights aligned to {len(gdf)} regions.")

    lisa = Moran_Local(gdf['mismatch'], w_clipped)
    gdf['lisa_I'] = lisa.Is
    gdf['lisa_p'] = lisa.p_sim
    gdf['lisa_q'] = lisa.q
    gdf['lisa_sig'] = (gdf['lisa_p'] < 0.05).astype(int)

    def classify(row):
        if row['lisa_sig'] == 0:
            return 'Not significant'
        return {
            1: 'High-High',
            2: 'Low-High',
            3: 'Low-Low',
            4: 'High-Low'
        }.get(row['lisa_q'], 'Not classified')

    gdf['lisa_type'] = gdf.apply(classify, axis=1)
    logging.info("LISA analysis completed.")
    return gdf

def plot_lisa_clusters(gdf, save_path="figures/lisa_clusters_map.png"):
    """Plot LISA cluster map.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with LISA results.
        save_path (str): Path to save the plot.
    """
    color_dict = {
        'High-High': 'red',
        'Low-Low': 'blue',
        'High-Low': 'yellowgreen',
        'Low-High': 'gold',
        'Not significant': 'lightgrey'
    }

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    patches = []
    for ctype, color in color_dict.items():
        subset = gdf[gdf['lisa_type'] == ctype]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor='black', linewidth=0.2)
            patches.append(plt.Rectangle((0, 0), 1, 1, fc=color, ec='none'))

    ax.set_title("LISA Cluster Map (Local Spatial Autocorrelation)", fontsize=14, fontweight='bold')
    if patches:
        ax.legend(
            patches, color_dict.keys(),
            title='Cluster Type', loc='lower left',
            handler_map={plt.Rectangle: HandlerPatch(
                patch_func=lambda legend, orig_handle, xdescent, ydescent, width, height, fontsize:
                plt.Rectangle(
                    (0, 0),
                    width,
                    height,
                    fc=orig_handle.get_facecolor(),
                    ec=None
                )
            )}
        )
    ax.axis('off')
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved LISA cluster map to {save_path}")
    plt.close()

# --- Spatial Lag Model ---

def compute_spatial_lag_model(df, gdf, save_path="figures/slm_coefficients.png"):
    """Compute Spatial Lag Model (SLM) to analyze mismatch spatial dependency.

    Args:
        df (pd.DataFrame): Enriched data with mismatch and control variables.
        gdf (gpd.GeoDataFrame): NUTS2 boundary data.
        save_path (str): Path to save the coefficient plot.

    Returns:
        ML_Lag: Spatial Lag Model results.
    """
    logging.info("Starting Spatial Lag Model analysis...")
    try:
        # Prepare data
        df_avg = df.groupby('region_code').agg({
            'mismatch': 'mean',
            'passenger_growth': 'mean',
            'pop_density': 'mean',
            'gdp_per_capita': 'mean'
        }).reset_index()
        logging.info(f"df_avg region_codes (top 5): {df_avg['region_code'].head().tolist()}")
        logging.info(f"gdf region_codes (top 5): {gdf['region_code'].head().tolist()}")

        # Exclude islands
        island_ids = [7, 20, 48, 52, 60, 61, 68, 70, 91, 94, 134, 156, 157, 169, 178, 179, 216, 219, 221, 230, 325]
        gdf = gdf[~gdf.index.isin(island_ids)]
        logging.info(f"Regions after island exclusion: {len(gdf)}")

        # Merge data
        gdf_merged = gdf.merge(df_avg, on='region_code', how='inner')
        gdf_merged = gdf_merged.dropna(subset=['mismatch', 'passenger_growth', 'pop_density', 'gdp_per_capita'])
        logging.info(f"Merged data: {len(gdf_merged)} regions")

        if gdf_merged.empty:
            logging.error("No data after merge and cleaning. Check region_code consistency.")
            return None

        # Create Queen contiguity weights
        w = Queen.from_dataframe(gdf_merged, use_index=True)
        w.transform = 'r'
        logging.info(f"Queen weights created for {len(gdf_merged)} regions.")

        # Prepare data for SLM
        y = gdf_merged['mismatch'].values
        X = gdf_merged[['passenger_growth', 'pop_density', 'gdp_per_capita']].values
        X_names = ['passenger_growth', 'pop_density', 'gdp_per_capita']

        # Run Spatial Lag Model
        slm = ML_Lag(y, X, w=w, name_y='mismatch', name_x=X_names)
        logging.info("\n📊 SLM regression results:")
        logging.info(slm.summary)

        # Extract coefficients and standard errors
        variables = ['CONSTANT'] + X_names + ['W_mismatch']
        coefficients = slm.betas.flatten().tolist()
        std_errors = slm.std_err.tolist()

        # Validate lengths
        logging.info(f"Variables: {len(variables)}, Coefficients: {len(coefficients)}, Std Errors: {len(std_errors)}")
        if not (len(variables) == len(coefficients) == len(std_errors)):
            logging.error(f"Length mismatch: Variables={len(variables)}, Coefficients={len(coefficients)}, Std Errors={len(std_errors)}")
            return slm

        # Create coefficient DataFrame
        coef_data = pd.DataFrame({
            'Variable': variables,
            'Coefficient': coefficients,
            'Std_Error': std_errors
        })

        # Visualize coefficients
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Variable', y='Coefficient', data=coef_data, color='skyblue')
        plt.errorbar(
            x=range(len(coef_data)),
            y=coef_data['Coefficient'],
            yerr=coef_data['Std_Error'],
            fmt='none', c='black', capsize=5
        )
        plt.title('Spatial Lag Model Coefficients')
        plt.xlabel('Variable')
        plt.ylabel('Coefficient Value')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved SLM coefficient plot to {save_path}")
        plt.close()

        return slm
    except Exception as e:
        logging.error(f"Error in SLM analysis: {e}")
        return None

# --- Main Execution ---

if __name__ == "__main__":
    try:
        df = load_enriched_data()
        gdf = load_nuts2_geojson()

        plot_average_mismatch_map(df, gdf, save_path="figures/average_mismatch_map.png")
        compute_morans_I(df, gdf)

        gdf_lisa = run_lisa_analysis(df, gdf)
        plot_lisa_clusters(gdf_lisa)

        # Run Spatial Lag Model
        compute_spatial_lag_model(df, gdf)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise