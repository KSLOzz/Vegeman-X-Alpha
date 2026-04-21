import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
import streamlit as st

@st.cache_data
def load_and_prepare_data(dfs):
    """
    Dynamically merges N years of DataFrames using a 15-meter metric threshold.
    Preserves Substation and Line Type metadata for GUI filtering.
    """
    # Force column names to lowercase and standardize lat/lon/line_type
    for df in dfs:
        df.columns = df.columns.str.strip().str.lower()
        df.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'line.type': 'line_type'}, inplace=True)
        
        # --- BRADLEY'S CLEARANCE FALLBACK LOGIC ---
        # If 'clearance' wasn't in the CSV, build it dynamically
        if 'clearance' not in df.columns:
            u_cols = [
                "encroachment.length.u0", "encroachment.length.u1",
                "encroachment.length.u2", "encroachment.length.u3",
                "encroachment.length.u4"
            ]
            clearance_map = [2.0, 5.0, 7.0, 9.0, 11.0]
            
            # Verify the encroachment columns actually exist before trying to use them
            if all(c in df.columns for c in u_cols):
                def compute_clearance(row):
                    u0 = row["encroachment.length.u0"]
                    if pd.isna(u0) or str(u0).strip().upper() == "NA":
                        return np.nan
                    for i, col in enumerate(u_cols):
                        try:
                            # float() handles any weird string numbers in the CSV
                            if float(row[col]) > 0:
                                return clearance_map[i]
                        except:
                            pass
                    return 0.0
                
                # Apply the function to create the missing column
                df["clearance"] = df.apply(compute_clearance, axis=1)
        # -----------------------------------------

        df.dropna(subset=['lat', 'lon'], inplace=True)


    # Use Year 1 (dfs[0]) as the Master Baseline
    df_master = dfs[0].copy()
    if 'clearance' in df_master.columns:
        df_master.rename(columns={'clearance': 'clearance_0'}, inplace=True)
        
    # Kellan's Conversion to GeoDataFrame and Project to Meters (EPSG:3857) for accurate distance math
    gdf_master = gpd.GeoDataFrame(
        df_master, 
        geometry=gpd.points_from_xy(df_master.lon, df_master.lat), 
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    
    # Loop through the rest of the years (Year 2, Year 3... Year N)
    for i in range(1, len(dfs)):
        # ONLY select the necessary columns and instantly drop any missing GPS coordinates
        df_next = dfs[i][['lat', 'lon', 'clearance']].rename(columns={'clearance': f'clearance_{i}'})
        df_next.dropna(subset=['lat', 'lon'], inplace=True)
        
        gdf_next = gpd.GeoDataFrame(
            df_next, 
            geometry=gpd.points_from_xy(df_next.lon, df_next.lat), 
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
        
        #  Drop redundant lat/lon text columns from Year 2 so they don't corrupt the Year 1 names
        gdf_next = gdf_next.drop(columns=['lat', 'lon'])
        
        # Spatial join connecting LiDAR hits within a strict 15-meter threshold
        gdf_master = gpd.sjoin_nearest(gdf_master, gdf_next, how='inner', max_distance=15, distance_col="dist")
        
        # Clean up spatial join artifacts to prep for the next loop
        cols_to_drop = [col for col in gdf_master.columns if col in ['index_right', 'dist']]
        gdf_master = gdf_master.drop(columns=cols_to_drop)
        
        # Drop duplicates to prevent the Cartesian memory explosion while respecting the 15m radius
        gdf_master = gdf_master.drop_duplicates(subset=['lat', 'lon'])

    # Convert back to a standard Pandas DataFrame for the math engine
    df_final = pd.DataFrame(gdf_master.drop(columns=['geometry']))
    
    # Drop any rows that missed clearance data across the timeline
    clearance_cols = [f'clearance_{i}' for i in range(len(dfs))]
    df_final = df_final.dropna(subset=clearance_cols)

    return df_final

def predict_and_cluster(df, historical_rains, rain_forecast, w_loc, w_growth, danger_threshold):
    """
    Calculates growth rates across N years, uses KNN to impute trimmed 
    clearances, predicts future clearance, and routes via K-Means.
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

    # 3. Isolate Natural Growth vs Trimmed
    df_train = df[df["r_point"] >= 0].copy()
    df_predict = df[df["r_point"] < 0].copy()

    # 4. RESTORED: Holdan's Nearest Neighbor (KNN) Imputation for trimmed vegetation
    if not df_predict.empty and not df_train.empty:
        # Use 5 nearest surrounding trees, weighted by distance, to predict localized growth rate
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
        knn.fit(df_train[['lat', 'lon']], df_train['r_point'])
        predicted_r = knn.predict(df_predict[['lat', 'lon']])
        
        df["r_point_imputed"] = df["r_point"]
        df.loc[df["r_point"] < 0, "r_point_imputed"] = predicted_r
    else:
        df["r_point_imputed"] = df["r_point"]

    # Clip top/bottom 1% to prevent insane math anomalies
    q_low, q_high = df["r_point_imputed"].quantile([0.01, 0.99])
    df["r_point_final"] = df["r_point_imputed"].clip(q_low, q_high)

    # 5. Predict Future Clearance
    latest_clearance_col = f"clearance_{num_intervals}"
    df["predicted_growth_final"] = df["r_point_final"] * rain_forecast
    df["predicted_clearance"] = df[latest_clearance_col] - df["predicted_growth_final"]

    # 6. Dynamic Safety Flagging
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

    return df
