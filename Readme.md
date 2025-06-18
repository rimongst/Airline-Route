# 🛫 EU Air Route Consistency Analysis (Master Thesis)

This repository contains the full codebase supporting my MSc thesis at EDHEC Business School.  
The project evaluates how well intra-EU airline routes respond to regional passenger demand, using multi-level modeling and predictive machine learning techniques.

## 📂 Project Structure

Master_Thesis/
├── data/ # Input & output data (not included in repo)
│ ├── figures/ # Visualizations and graphs
│ └── output/ # Model and analysis results
├── scripts/ # All functional analysis scripts
├── main.py # Entry point to coordinate the pipeline
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
└── README.md # This project overview

# ⚠️ Project Status
This repository is a work in progress. Some components and scripts are still under revision as part of ongoing thesis development.

Please note:

Data files are not uploaded due to size or privacy concerns.

Future plans include replacing local data sources with dynamic API-based imports for better automation and reproducibility.

## 🔧 Key Scripts

- **`Clean_project.py`**: General data cleaning pipeline for the project.
- **`main.py`**: Main entry point to orchestrate all modules and workflows.
- **`missingvalue_analyse.py`**: Analyzes and visualizes missing values in the dataset.
- **`Passenger_Clean.py`**: Processes Eurostat passenger volume data.
- **`Prediction.py`**: Builds and evaluates prediction models for route continuation using ML.
- **`Prepare.py`**: Merges route and passenger growth datasets, handles missing values.
- **`RegionLevel_Consistency.py`**: Analyzes route-passenger consistency at the NUTS2 regional level.
- **`RouteLevel_Clean.py`**: Filters and processes raw route-level datasets.
- **`RouteLevel_Consistency.py`**: Implements ±1 consistency score logic for each route-year.
- **`SpatialLevel_Analysis.py`**: Conducts spatial autocorrelation analysis (Moran’s I, LISA).

## 📊 Methodology Summary

- Construction of ±1 consistency score per route-year  
- Regional-level mismatch rate (passenger growth vs route supply)  
- Spatial clustering using Moran’s I and LISA  
- Predictive modeling using Logistic Regression, Random Forest, XGBoost, and GCN  

## 🚀 Usage

1. Clone the repository and install requirements:
   ```bash
   pip install -r requirements.txt

2. Run modules individually, e.g.:
   ```bash
   python scripts/Prediction.py
   
# 📚 Thesis Context

This code supports the thesis:
Structure-Aware Consistency of EU Air Routes: Multi-Level Analysis & Predictive Modeling

# 🧠 Author

Shitan Gao

EDHEC MSc in Data Analytics & Artificial Intelligence

Shitan.gao@edhec.com

