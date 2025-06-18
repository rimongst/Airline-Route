import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Set random seed for reproducibility
np.random.seed(42)


def haversine_np(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in kilometers."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def merge_parquet_to_csv(parquet_paths):
    """Merge multiple Parquet files into a single DataFrame."""
    df_list = []
    for p in parquet_paths:
        if not Path(p).is_file():
            logging.error(f"File not found: {p}")
            continue
        try:
            df_list.append(pd.read_parquet(p))
        except Exception as e:
            logging.error(f"Failed to read {p}: {e}")
    if not df_list:
        raise ValueError("No valid Parquet files loaded")
    df_all = pd.concat(df_list, ignore_index=True)
    df_all['year'] = df_all['year'].astype(int)
    logging.info(f"Merged {len(df_all)} rows from {len(df_list)} files")
    return df_all


def generate_route_level_consistency(df, output_full_path, output_summary_path=None,
                                     plot_path="figures/route_consistency_score_trend.png"):
    """Compute ±1 consistency score and save results."""
    working_df = df[['origin_icao', 'dest_icao', 'year', 'value']].copy()
    initial_rows = len(working_df)
    working_df = working_df.dropna()
    dropped_rows = initial_rows - len(working_df)
    if dropped_rows > 0:
        logging.warning(f"Dropped {dropped_rows} rows with missing values ({dropped_rows / initial_rows:.2%})")
        working_df[working_df.isna().any(axis=1)][['origin_icao', 'dest_icao', 'year']].to_csv(
            "output/missing_route_values.csv", index=False)
        logging.info("Saved missing value rows to output/missing_route_values.csv")

    working_df.rename(columns={
        'origin_icao': 'origin_airport',
        'dest_icao': 'destination_airport',
        'value': 'passenger_count'
    }, inplace=True)

    # Validate ICAO codes
    invalid_icao = working_df[~working_df['origin_airport'].str.match(r'^[A-Z]{4}$') |
                              ~working_df['destination_airport'].str.match(r'^[A-Z]{4}$')]
    if not invalid_icao.empty:
        logging.warning(f"Found {len(invalid_icao)} rows with invalid ICAO codes")
        invalid_icao.to_csv("output/invalid_icao_codes.csv", index=False)
        working_df = working_df[working_df['origin_airport'].str.match(r'^[A-Z]{4}$') &
                                working_df['destination_airport'].str.match(r'^[A-Z]{4}$')]

    working_df['route_id'] = working_df['origin_airport'] + '_' + working_df['destination_airport']
    working_df = working_df.sort_values(by=['route_id', 'year'])
    working_df['passenger_growth'] = working_df.groupby('route_id')['passenger_count'].pct_change()
    working_df['passenger_growth'] = working_df['passenger_growth'].clip(lower=-3, upper=3)

    working_df['exists_next_year'] = working_df.groupby('route_id')['year'].shift(-1).notnull().astype(int)

    # Enhanced consistency score with growth magnitude
    working_df['consistency_score'] = 0
    working_df.loc[
        (working_df['passenger_growth'] > 0.5) & (working_df['exists_next_year'] == 1), 'consistency_score'] = 2
    working_df.loc[
        (working_df['passenger_growth'] > 0) & (working_df['exists_next_year'] == 1), 'consistency_score'] = 1
    working_df.loc[
        (working_df['passenger_growth'] < -0.5) & (working_df['exists_next_year'] == 0), 'consistency_score'] = 2
    working_df.loc[
        (working_df['passenger_growth'] < 0) & (working_df['exists_next_year'] == 0), 'consistency_score'] = 1
    working_df.loc[((working_df['passenger_growth'] > 0) & (working_df['exists_next_year'] == 0)) |
                   ((working_df['passenger_growth'] < 0) & (
                               working_df['exists_next_year'] == 1)), 'consistency_score'] = -1

    output_dir = Path(output_full_path).parent
    output_dir.mkdir(exist_ok=True)
    working_df.to_parquet(output_full_path, index=False)

    if output_summary_path:
        summary = working_df.groupby('year')['consistency_score'].value_counts(normalize=True).unstack().fillna(0)
        summary.to_parquet(output_summary_path)

        plt.figure(figsize=(12, 6))
        ax = summary.plot(kind='bar', stacked=True, colormap='viridis')
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f%%', label_type='edge', fontsize=8)
        plt.title("Annual Route Consistency Score as a Percentage")
        plt.xlabel("Year")
        plt.ylabel("Percentage")
        plt.legend(title="Consistency Score", loc="upper right")
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(plot_path, dpi=300)
        logging.info(f"Saved stacked bar chart to {plot_path}")
        plt.close()

    logging.info(f'Route-level consistency computed and saved to: {output_full_path}')
    return working_df


def analyze_consistency_by_route_type(df, df_with_coords, distance_thresholds=[500, 2500],
                                      save_path="figures/consistency_short_vs_long.png"):
    """Compare consistency score trends for short-haul vs. long-haul routes with multiple distance thresholds."""
    df = df.copy()
    coords = df_with_coords[
        ['origin_icao', 'dest_icao', 'origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']].drop_duplicates()
    df = df.merge(coords, left_on=['origin_airport', 'destination_airport'],
                  right_on=['origin_icao', 'dest_icao'], how='left')

    df = df[df['origin_lat'].notna() & df['dest_lat'].notna()]
    df['distance_km'] = haversine_np(df['origin_lat'], df['origin_lon'], df['dest_lat'], df['dest_lon'])
    df = df[df['distance_km'].notna() & df['consistency_score'].notna()]

    output_dir = Path(save_path).parent
    output_dir.mkdir(exist_ok=True)

    # Save summary statistics for each threshold
    summary_stats = []

    for threshold in distance_thresholds:
        # Classify routes based on distance
        df['route_type'] = np.where(df['distance_km'] < threshold, f'Short (<{threshold}km)', f'Long (≥{threshold}km)')

        # Compute mean consistency score by year and route type
        grouped = df.groupby(['year', 'route_type'])['consistency_score'].mean().unstack().fillna(0)

        # Statistical summary
        stats = df.groupby('route_type')['consistency_score'].agg(['mean', 'std']).reset_index()
        stats['threshold'] = threshold
        summary_stats.append(stats)

        # Plotting
        plt.figure(figsize=(12, 6))
        for rt in grouped.columns:
            plt.plot(grouped.index, grouped[rt], label=rt, marker='o')
        plt.title(f'Mean Consistency Score: Short vs. Long-Haul Routes (Threshold: {threshold}km)')
        plt.xlabel('Year')
        plt.ylabel('Mean Consistency Score')
        plt.legend(title='Route Type')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_path_threshold = save_path.replace('.png', f'_{threshold}km.png')
        plt.savefig(save_path_threshold, dpi=300)
        logging.info(f"Saved consistency plot for {threshold}km to {save_path_threshold}")
        plt.close()

    # Save summary statistics to CSV
    summary_stats_df = pd.concat(summary_stats, ignore_index=True)
    summary_stats_path = output_dir / "consistency_stats_by_threshold.csv"
    summary_stats_df.to_csv(summary_stats_path, index=False)
    logging.info(f"Saved summary statistics to {summary_stats_path}")


def find_typical_inconsistent_routes(df, top_n=15, growth_threshold=0.25):
    """Identify typical inconsistent routes with high growth but canceled."""
    inconsistent_df = df[
        (df['consistency_score'] == -1) &
        (df['passenger_growth'] > growth_threshold) &
        (df['exists_next_year'] == 0)
        ].copy()

    inconsistent_df = inconsistent_df.sort_values(by='passenger_growth', ascending=False)
    logging.info(f"Found {len(inconsistent_df)} inconsistent routes with high growth but canceled")
    return inconsistent_df.head(top_n)


def plot_growth_vs_retention(df, save_path="figures/growth_vs_retention.png"):
    """Plot average passenger growth vs. route retention rate per year."""
    df = df.copy()
    df = df[df['passenger_growth'].notna()]
    df['passenger_growth'] = df['passenger_growth'].clip(lower=-3, upper=3)
    df = df[df['year'] < df['year'].max()]  # Exclude incomplete current year

    yearly_stats = df.groupby('year').agg({
        'exists_next_year': 'mean',
        'passenger_growth': 'mean'
    }).rename(columns={
        'exists_next_year': 'route_retention_rate',
        'passenger_growth': 'avg_passenger_growth'
    })

    scaler = MinMaxScaler()
    yearly_stats[['route_retention_rate', 'avg_passenger_growth']] = scaler.fit_transform(
        yearly_stats[['route_retention_rate', 'avg_passenger_growth']])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(yearly_stats.index, yearly_stats['route_retention_rate'], label='Route Retention Rate', color='tab:blue')
    ax1.set_ylabel('Normalized Route Retention Rate', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(yearly_stats.index, yearly_stats['avg_passenger_growth'], label='Avg Passenger Growth', color='tab:green')
    ax2.set_ylabel('Normalized Avg Passenger Growth', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title("Passenger Growth vs Route Retention Trend (Normalized)")
    fig.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path, dpi=300)
    logging.info(f"Saved passenger vs retention trend figure to {save_path}")
    plt.close()


def run_extended_logistic_model(df_consistency, df_routes, top_airports=None,
                                save_path="figures/extended_logistic_model_prediction.png"):
    """Run logistic regression model to predict route retention."""
    df = df_consistency.copy()
    coords = df_routes[['origin_icao', 'dest_icao', 'origin_lat', 'origin_lon',
                        'dest_lat', 'dest_lon', 'origin_country', 'dest_country']].drop_duplicates()
    df = df.merge(coords, left_on=['origin_airport', 'destination_airport'],
                  right_on=['origin_icao', 'dest_icao'], how='left')

    df = df[df['origin_lat'].notna() & df['dest_lat'].notna()]
    df['distance_km'] = haversine_np(df['origin_lat'], df['origin_lon'], df['dest_lat'], df['dest_lon'])
    df['is_international'] = (df['origin_country'] != df['dest_country']).astype(int)

    if top_airports is not None:
        df['is_major_origin'] = df['origin_airport'].isin(top_airports).astype(int)
        df['is_major_dest'] = df['destination_airport'].isin(top_airports).astype(int)
    else:
        df['is_major_origin'] = 0
        df['is_major_dest'] = 0

    feature_cols = ['passenger_growth', 'distance_km', 'is_international', 'is_major_origin', 'is_major_dest']
    df_model = df[['exists_next_year'] + feature_cols].copy()
    df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

    logging.info(f"Modeling sample size: {len(df_model)}")
    logging.info("Missing value counts:\n" + str(df_model.isna().sum()))

    X = df_model[feature_cols]
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X[feature_cols]), columns=feature_cols, index=X.index)
    X_scaled = sm.add_constant(X_scaled)
    y = df_model['exists_next_year']

    # Check multicollinearity
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_scaled.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]
    logging.info("Variance Inflation Factors:\n" + str(vif_data))

    model = sm.Logit(y, X_scaled).fit(disp=0)
    logging.info(model.summary())

    df_model['predicted_prob'] = model.predict(X_scaled)
    plt.figure(figsize=(10, 5))
    plt.hist(df_model[df_model['exists_next_year'] == 1]['predicted_prob'], bins=30, alpha=0.6, label='Retained')
    plt.hist(df_model[df_model['exists_next_year'] == 0]['predicted_prob'], bins=30, alpha=0.6, label='Cancelled')
    plt.xlabel('Predicted Probability of Route Retention')
    plt.ylabel('Count')
    plt.title('Extended Logistic Model Prediction Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logging.info(f"Saved logistic model prediction plot to {save_path}")
    plt.close()

    return model


if __name__ == "__main__":
    df_routes = merge_parquet_to_csv([
        "output/routes_2000s.parquet",
        "output/routes_2010s.parquet",
        "output/routes_2020s.parquet"
    ])

    df_consistency = generate_route_level_consistency(
        df=df_routes,
        output_full_path="output/route_consistency.parquet",
        output_summary_path="output/route_consistency_summary.parquet",
        plot_path="figures/route_consistency_score_trend.png"
    )

    analyze_consistency_by_route_type(
        df_consistency,
        df_routes,
        distance_thresholds=[500, 2500],  # Updated thresholds
        save_path="figures/consistency_short_vs_long.png"
    )

    top_cases = find_typical_inconsistent_routes(df_consistency, top_n=15, growth_threshold=0.25)
    logging.info("Top inconsistent routes:\n" + str(top_cases))

    plot_growth_vs_retention(df_consistency, save_path="figures/growth_vs_retention.png")

    top10 = ['EHAM', 'LFPG', 'EDDF', 'LGAV', 'LEMD', 'LIRF', 'LEPA', 'EDDM', 'LPPT', 'LEBL']
    run_extended_logistic_model(
        df_consistency,
        df_routes,
        top_airports=top10,
        save_path="figures/extended_logistic_model_prediction.png"
    )