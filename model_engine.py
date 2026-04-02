import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import statsmodels.formula.api as smf
import streamlit as st

@st.cache_data
def load_and_prepare_data(dfs):
    """
    Dynamically merges N years of DataFrames, using the first year as the spatial baseline.
    Preserves Substation and Line Type metadata for GUI filtering.
    """
    # Force column names to lowercase and standardize lat/lon/line_type
    for df in dfs:
        df.columns = df.columns.str.strip().str.lower()
        # This line intercepts 'latitude', 'longitude', and 'line.type' and standardizes them
        df.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'line.type': 'line_type'}, inplace=True)

    # Use Year 1 (dfs[0]) as the Master Baseline
    df_master = dfs[0].copy()
    
    # Rename the first year's clearance column
    if 'clearance' in df_master.columns:
        df_master = df_master.rename(columns={'clearance': 'clearance_0'})
    
    # Loop through the rest of the years (Year 2, Year 3... Year N)
    for i in range(1, len(dfs)):
        df_next = dfs[i]
        df_next = df_next[['lat', 'lon', 'clearance']].rename(columns={'clearance': f'clearance_{i}'})
        
        # Staple it to the master dataframe matching exactly on GPS coordinates
        df_master = pd.merge(df_master, df_next, on=['lat', 'lon'], how='inner')

    # Drop any rows that have missing clearance data across the timeline
    clearance_cols = [f'clearance_{i}' for i in range(len(dfs))]
    df_master = df_master.dropna(subset=clearance_cols)

    return df_master

def predict_and_cluster(df, historical_rains, rain_forecast, w_loc, w_growth, danger_threshold):
    """
    Calculates growth rates across N years, applies the LMM, 
    predicts future clearance, and routes via K-Means.
    """
    num_intervals = len(historical_rains)
    r_columns = []
    
    # 1. Calculate dynamic growth and R-values for every historical interval
    for i in range(num_intervals):
        df[f"growth_{i}"] = df[f"clearance_{i}"] - df[f"clearance_{i+1}"]
        df[f"r_{i}"] = df[f"growth_{i}"] / historical_rains[i]
        r_columns.append(f"r_{i}")

    # 2. Average all the individual yearly R-values into a single baseline point
    df["r_point"] = df[r_columns].mean(axis=1)

    # 3. Micro-Climate Grids (Random Effect Groups) for LMM
    df["grid_id"] = df["lat"].round(1).astype(str) + "_" + df["lon"].round(1).astype(str)

    # 4. Isolate Natural Growth vs Trimmed
    df_train = df[df["r_point"] >= 0].copy()
    df_predict = df[df["r_point"] < 0].copy()

    # 5. Fit the Mixed-Effects Model
    try:
        md = smf.mixedlm("r_point ~ clearance_0", df_train, groups="grid_id")
        mdf = md.fit()
        predicted_r = mdf.predict(df_predict)
        
        df["r_point_imputed"] = df["r_point"]
        df.loc[df["r_point"] < 0, "r_point_imputed"] = predicted_r
    except Exception as e:
        print(f"LMM failed to converge: {e}")
        overall_mean = df_train["r_point"].mean()
        df["r_point_imputed"] = df["r_point"].apply(lambda x: overall_mean if x < 0 else x)

    # Clip top/bottom 1% to prevent insane math anomalies
    q_low, q_high = df["r_point_imputed"].quantile([0.01, 0.99])
    df["r_point_final"] = df["r_point_imputed"].clip(q_low, q_high)

    # 6. Predict Future Clearance
    latest_clearance_col = f"clearance_{num_intervals}"
    df["predicted_growth_final"] = df["r_point_final"] * rain_forecast
    df["predicted_clearance"] = df[latest_clearance_col] - df["predicted_growth_final"]

    # 7. Dynamic Safety Flagging
    conditions = [
        (df["predicted_clearance"] < danger_threshold),
        (df["predicted_clearance"] >= danger_threshold) & (df["predicted_clearance"] < danger_threshold + 2),
        (df["predicted_clearance"] >= danger_threshold + 2) & (df["predicted_clearance"] < danger_threshold + 4),
        (df["predicted_clearance"] >= danger_threshold + 4) & (df["predicted_clearance"] < 10),
        (df["predicted_clearance"] >= 10)
    ]
    colors = ['#ff0000', '#ffa500', '#ffff00', '#008000', '#ffffff'] # Red, Orange, Yellow, Green, White
    buckets = ['Critical', 'High Priority', 'Watch List', 'Healthy', 'Safe']
    
    df['risk_color'] = np.select(conditions, colors, default='#ffffff')
    df['risk_bucket'] = np.select(conditions, buckets, default='Safe')

    # 8. K-Means Routing Algorithm
    # Only route the critical/high priority trees
    df_dangerous = df[df['predicted_clearance'] <= (danger_threshold + 2)].copy()
    
    if not df_dangerous.empty:
        lat_std = (df_dangerous['lat'] - df_dangerous['lat'].mean()) / df_dangerous['lat'].std()
        lon_std = (df_dangerous['lon'] - df_dangerous['lon'].mean()) / df_dangerous['lon'].std()
        growth_std = (df_dangerous['predicted_growth_final'] - df_dangerous['predicted_growth_final'].mean()) / df_dangerous['predicted_growth_final'].std()

        X_cluster = np.column_stack((
            lat_std * w_loc, 
            lon_std * w_loc, 
            growth_std * w_growth
        ))

        n_clusters = min(5, len(df_dangerous))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_dangerous['dispatch_cluster'] = kmeans.fit_predict(X_cluster)
        
        df = df.merge(df_dangerous[['lat', 'lon', 'dispatch_cluster']], on=['lat', 'lon'], how='left')
    else:
        df['dispatch_cluster'] = np.nan

    return df
