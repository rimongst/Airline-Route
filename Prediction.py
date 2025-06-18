import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import gc
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


def load_main_data(sample_frac: float = 0.01, target: str = "consistency_score") -> pd.DataFrame:
    """Load main route-level data with optional sampling."""
    data_path = Path("output/enriched_route_level.parquet")
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_parquet(data_path, engine='pyarrow')

    required_cols = ["origin_airport", "destination_airport", "year", target, "passenger_growth", "passenger_count"]
    if not all(col in df.columns for col in required_cols):
        logging.error(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {required_cols}")

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        logging.info(f"Sampled {sample_frac:.0%} of main data → {len(df)} rows")

    # Feature engineering
    df['passenger_growth'] = np.log1p(df['passenger_growth'].clip(lower=0, upper=df['passenger_growth'].quantile(0.99)))
    df['passenger_count'] = np.log1p(df['passenger_count'].clip(lower=0, upper=df['passenger_count'].quantile(0.99)))

    # Check distributions
    logging.info(f"Target distribution: {df[target].describe().to_dict()}")
    plt.figure()
    df[target].hist()
    plt.title("Consistency Score Distribution")
    plt.savefig("output/consistency_score_hist.png")
    plt.close()
    for col in ['passenger_growth', 'passenger_count']:
        logging.info(f"{col} distribution: {df[col].describe().to_dict()}")
        plt.figure()
        df[col].hist()
        plt.title(f"{col} Distribution")
        plt.savefig(f"output/{col}_hist.png")
        plt.close()

    df[target] = (df[target] - df[target].mean()) / df[target].std()  # Normalize target
    logging.info(f"Data columns: {list(df.columns)}")
    gc.collect()
    return df


def load_airport_facilities() -> pd.DataFrame:
    """Load and process airport facilities data."""
    facilities_path = Path("data/estat_avia_if_typ.tsv")
    if not facilities_path.is_file():
        raise FileNotFoundError(f"Facilities file not found: {facilities_path}")
    df_if = pd.read_csv(facilities_path, sep="\t")
    df_if = df_if.rename(columns={df_if.columns[0]: "key"})
    df_if[["freq", "tra_infr", "rep_airp"]] = df_if["key"].str.split(",", expand=True)
    df_if = df_if.drop(columns=["key", "freq"])

    # Debug rep_airp format
    logging.info(f"Sample rep_airp values: {df_if['rep_airp'].sample(5).to_list()}")

    df_if = df_if.melt(id_vars=["tra_infr", "rep_airp"], var_name="year", value_name="value")
    df_if["year"] = df_if["year"].str.extract("(\d{4})").astype(int)
    df_if["value"] = pd.to_numeric(df_if["value"], errors="coerce")
    df_if["icao_airport"] = df_if["rep_airp"].str.extract(r"([A-Z]{4})")

    df_if = df_if.pivot_table(index=["icao_airport", "year"], columns="tra_infr", values="value").reset_index()
    df_if.columns.name = None
    logging.info(f"Loaded facilities data with {len(df_if)} rows")
    gc.collect()
    return df_if


def merge_facilities(df_route: pd.DataFrame, df_if: pd.DataFrame) -> pd.DataFrame:
    """Merge route data with airport facilities."""
    df = df_route.merge(
        df_if,
        left_on=["origin_airport", "year"],
        right_on=["icao_airport", "year"],
        how="left"
    )
    missing = df["icao_airport"].isna().sum()
    if missing > 0:
        total = len(df)
        missing_ratio = missing / total
        unmatched_airports = df[df["icao_airport"].isna()]["origin_airport"].value_counts().head(10)
        unmatched_years = df[df["icao_airport"].isna()]["year"].value_counts().head(5)
        logging.warning(
            f"{missing} rows ({missing_ratio:.2%}) with unmatched origin_airport. Top unmatched airports: {unmatched_airports}, Top years: {unmatched_years}")

    df = df.rename(columns={
        "RWAY": "runway_count",
        "PAS_GATE": "gate_count",
        "CKIN": "checkin_desks"
    })
    facility_cols = ["runway_count", "gate_count", "checkin_desks"]
    for col in facility_cols:
        df[col] = df.groupby("origin_airport")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(0)
    logging.info(f"Facility columns missing rates: {df[facility_cols].isna().mean().to_dict()}")
    gc.collect()
    return df.drop(columns=["icao_airport", "rep_airp"], errors="ignore")


def train_and_evaluate_models(
        df_model: pd.DataFrame,
        features: list,
        target: str = "consistency_score",
        sample_frac: float = 0.1,
        classification: bool = False
) -> Tuple[Optional[RandomForestRegressor], Optional[XGBRegressor], Optional[pd.DataFrame]]:
    """Train and evaluate regression or classification models with SHAP analysis."""
    start_time = time.time()
    try:
        df_model = df_model.dropna(subset=[target] + features)

        if sample_frac < 1.0:
            df_model = df_model.sample(frac=sample_frac, random_state=42)
            logging.info(f"⚡ Sampled {sample_frac:.0%} → {len(df_model)} rows")

        X = df_model[features]
        y = df_model[target]
        if classification:
            y = (y > 0).astype(int)  # Binary classification: consistency_score > 0
            logging.info(f"✅ Classification mode | Data size: {len(df_model)} | Target positive rate: {y.mean():.4f}")
        else:
            logging.info(f"✅ Regression mode | Data size: {len(df_model)} | Target summary: {y.describe().to_dict()}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=features, index=df_model.index)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        def evaluate_model(name: str, model):
            if classification:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                logging.info(f"\n📊 {name} Evaluation (Classification):")
                logging.info(f"Accuracy: {accuracy:.4f}")
                logging.info(f"Classification Report: {report}")
                return accuracy
            else:
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                logging.info(f"\n📊 {name} Evaluation (Regression):")
                logging.info(f"RMSE: {rmse:.4f}")
                logging.info(f"R2 Score: {r2:.4f}")
                return rmse, r2

        # Linear Regression
        linear = None
        try:
            start = time.time()
            linear = LinearRegression()
            linear.fit(X_train, y_train)
            logging.info(f"Linear Regression training time: {time.time() - start:.2f}s")
            evaluate_model("Linear Regression", linear)
        except Exception as e:
            logging.error(f"Linear Regression failed: {e}")

        # Random Forest
        rf = None
        try:
            start = time.time()
            rf_grid = GridSearchCV(
                RandomForestRegressor(random_state=42) if not classification else RandomForestClassifier(
                    random_state=42),
                param_grid={'n_estimators': [50], 'max_depth': [10]},
                scoring='neg_mean_squared_error' if not classification else 'accuracy',
                cv=3,
                n_jobs=1,
                verbose=0
            )
            rf_grid.fit(X_train, y_train)
            rf = rf_grid.best_estimator_
            logging.info(f"Best RF params: {rf_grid.best_params_}, training time: {time.time() - start:.2f}s")
            evaluate_model("Random Forest", rf)
        except Exception as e:
            logging.error(f"Random Forest failed: {e}")

        # XGBoost
        xgb = None
        try:
            start = time.time()
            xgb_grid = GridSearchCV(
                XGBRegressor(random_state=42) if not classification else XGBClassifier(random_state=42),
                param_grid={'n_estimators': [50], 'max_depth': [3], 'learning_rate': [0.1]},
                scoring='neg_mean_squared_error' if not classification else 'accuracy',
                cv=3,
                n_jobs=1,
                verbose=0
            )
            xgb_grid.fit(X_train, y_train)
            xgb = xgb_grid.best_estimator_
            logging.info(f"Best XGB params: {xgb_grid.best_params_}, training time: {time.time() - start:.2f}s")
            evaluate_model("XGBoost", xgb)
        except Exception as e:
            logging.error(f"XGBoost failed: {e}")

        gc.collect()
        logging.info(f"Total training time: {time.time() - start_time:.2f}s")
        return rf, xgb, X_test

    except Exception as e:
        logging.error(f"train_and_evaluate_models failed: {e}")
        return None, None, None


def plot_feature_importance(rf: Optional[RandomForestRegressor], features: list):
    """Plot feature importance from Random Forest."""
    if rf is None:
        logging.warning("Random Forest model is None, skipping feature importance plot")
        return
    try:
        feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values()
        plt.figure(figsize=(8, 6))
        sns.barplot(x=feat_imp.values, y=feat_imp.index)
        plt.title("Feature Importance (Random Forest)")
        plt.tight_layout()
        plt.savefig("output/feature_importance.png", dpi=300)
        plt.close()
        gc.collect()
    except Exception as e:
        logging.error(f"Feature importance plot failed: {e}")


def plot_shap_analysis(model, model_name: str, X_test: pd.DataFrame, features: list):
    """Plot SHAP analysis for a given model."""
    try:
        if model is None:
            logging.warning(f"{model_name} model is None, skipping SHAP analysis")
            return

        # Determine explainer based on model type
        if isinstance(model, LinearRegression):
            explainer = shap.Explainer(model, X_test)
        else:
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_test)

        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
        plt.title(f"SHAP Summary for {model_name}")
        plt.savefig(f"output/shap_{model_name.lower().replace(' ', '_')}_summary.png")
        plt.close()

        # Dependence plot for top feature
        top_feature = features[np.argmax(np.abs(shap_values).mean(0))]
        plt.figure()
        shap.dependence_plot(top_feature, shap_values, X_test, feature_names=features, show=False)
        plt.title(f"SHAP Dependence Plot for {top_feature} ({model_name})")
        plt.savefig(f"output/shap_{model_name.lower().replace(' ', '_')}_dependence.png")
        plt.close()

        logging.info(f"SHAP analysis saved for {model_name}")
        gc.collect()
    except Exception as e:
        logging.error(f"SHAP analysis failed for {model_name}: {e}")


def create_graph_data(df: pd.DataFrame, features: list, year: int, target: str = "consistency_score") -> Data:
    """Create PyTorch Geometric graph data for a specific year."""
    df_year = df[df['year'] == year].sample(min(len(df[df['year'] == year]), 2000), random_state=42)
    if df_year.empty:
        logging.warning(f"No data for year {year}")
        return None

    # Preprocess passenger_growth for edge weights
    df_year['passenger_growth'] = df_year['passenger_growth'].apply(
        lambda x: max(min(x, 5), -5) if not np.isnan(x) else 0)

    airports = pd.concat([df_year['origin_airport'], df_year['destination_airport']]).unique()
    airport_to_idx = {airport: idx for idx, airport in enumerate(airports)}

    node_features = []
    node_labels = []
    for airport in airports:
        airport_data = df_year[df_year['origin_airport'] == airport][features].mean()
        if airport_data.isna().all():
            airport_data = df_year[df_year['destination_airport'] == airport][features].mean()
        node_features.append(airport_data.fillna(0).values)

        routes = df_year[(df_year['origin_airport'] == airport) | (df_year['destination_airport'] == airport)]
        label = routes[target].mean() if not routes.empty else 0
        node_labels.append(label)

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.float)

    edge_index = []
    edge_weights = []
    for _, row in df_year.iterrows():
        src = airport_to_idx[row['origin_airport']]
        dst = airport_to_idx[row['destination_airport']]
        edge_index.append([src, dst])
        edge_index.append([dst, src])  # Undirected graph
        weight = row['passenger_growth']
        edge_weights.extend([weight, weight])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y=y)
    logging.info(f"Graph for year {year}: {len(airports)} nodes, {edge_index.shape[1]} edges")
    gc.collect()
    return data


class GCN(torch.nn.Module):
    """Graph Convolutional Network for regression."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.fc(x)
        return x.squeeze()


def train_gnn_model(df: pd.DataFrame, features: list, target: str = "consistency_score", output_dir: str = "output"):
    """Train and evaluate GCN model without SHAP analysis."""
    logging.info("Starting GNN training...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    logging.info(f"Using device: {device}")

    years = sorted(df['year'].unique())
    train_years = years[:-2]
    test_year = years[-1]

    train_graphs = [create_graph_data(df, features, year, target) for year in train_years]
    train_graphs = [g for g in train_graphs if g is not None]
    test_graph = create_graph_data(df, features, test_year, target)

    if not train_graphs or test_graph is None:
        logging.error("Insufficient graph data for GNN training")
        return None

    model = GCN(in_channels=len(features), hidden_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(50):
        total_loss = 0
        for data in train_graphs:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}, Loss: {total_loss / len(train_graphs):.4f}")

    model.eval()
    with torch.no_grad():
        test_graph = test_graph.to(device)
        out = model(test_graph)
        y_pred = out.cpu().numpy()
        y_true = test_graph.y.cpu().numpy()
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        logging.info("\n📊 GNN Evaluation (Regression):")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"R2 Score: {r2:.4f}")

    pred_df = pd.DataFrame({
        'airport_idx': range(len(y_true)),
        'y_true': y_true,
        'y_pred': y_pred
    })
    pred_df.to_csv(output_path / f"gnn_predictions_{target}.csv", index=False)
    logging.info(f"GNN predictions saved to {output_path / f'gnn_predictions_{target}.csv'}")
    gc.collect()
    return model


def plot_correlation_matrix(df: pd.DataFrame, features: list, target: str):
    """Plot correlation matrix."""
    try:
        corr = df[features + [target]].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig("output/correlation_matrix.png", dpi=300)
        plt.close()
        logging.info("Correlation matrix saved to output/correlation_matrix.png")
    except Exception as e:
        logging.error(f"Correlation matrix plot failed: {e}")


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    try:
        target = "consistency_score"
        features = ["passenger_growth", "passenger_count"]

        df_route = load_main_data(sample_frac=0.01, target=target)
        df_if = load_airport_facilities()
        df_merged = merge_facilities(df_route, df_if)

        plot_correlation_matrix(df_merged, features, target)

        # Try regression with SHAP
        rf, xgb, X_test = train_and_evaluate_models(
            df_model=df_merged,
            features=features,
            target=target,
            sample_frac=0.1,
            classification=False
        )

        if rf:
            plot_feature_importance(rf, features)
            plot_shap_analysis(rf, "Random Forest", X_test, features)
        if xgb:
            plot_shap_analysis(xgb, "XGBoost", X_test, features)

        # Try classification
        rf_clf, xgb_clf, X_test_clf = train_and_evaluate_models(
            df_model=df_merged,
            features=features,
            target=target,
            sample_frac=0.1,
            classification=True
        )

        if rf_clf:
            plot_shap_analysis(rf_clf, "Random Forest (Classification)", X_test_clf, features)
        if xgb_clf:
            plot_shap_analysis(xgb_clf, "XGBoost (Classification)", X_test_clf, features)

        gnn_model = train_gnn_model(df_merged, features, target=target, output_dir="output")
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        raise