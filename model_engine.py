import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ====================================================================
# FUNCTION 1: THE HEAVY DATA PROCESSING (Runs once when files upload)
# ====================================================================
def load_and_prepare_data(file_22, file_23, file_24):
    """Executes Steps 0 through 5 of Vegeman2_1.py in memory."""
    
    # --- STEP 0: Compute Clearance for 2022 ---
    df_2022_raw = pd.read_csv(file_22)
    U_COLS = ["encroachment.length.u0", "encroachment.length.u1", "encroachment.length.u2", "encroachment.length.u3", "encroachment.length.u4"]
    CLEARANCE_MAP = [2.0, 5.0, 7.0, 9.0, 11.0]

    def compute_clearance(row):
        u0 = row.get("encroachment.length.u0", np.nan)
        if pd.isna(u0) or str(u0).strip().upper() == "NA":
            return np.nan
        for i, col in enumerate(U_COLS):
            if col in row and float(row[col]) > 0:
                return CLEARANCE_MAP[i]
        return 0.0

    # Only apply if it's the raw format missing clearance
    if "Clearance" not in df_2022_raw.columns and "encroachment.length.u0" in df_2022_raw.columns:
        df_2022_raw["Clearance"] = df_2022_raw.apply(compute_clearance, axis=1)

    # --- STEP 1: Load and Rename ---
    cols_to_use = ['Latitude', 'Longitude', 'Clearance']
    
    # Read remaining files
    df_2023_raw = pd.read_csv(file_23, usecols=cols_to_use)
    df_2024_raw = pd.read_csv(file_24, usecols=cols_to_use)

    # Filter columns for 2022 just in case it has extra junk
    df_2022 = df_2022_raw[cols_to_use].rename(columns={'Latitude':'lat_2022','Longitude':'lon_2022','Clearance':'clearance_2022'})
    df_2023 = df_2023_raw.rename(columns={'Latitude':'lat_2023','Longitude':'lon_2023','Clearance':'clearance_2023'})
    df_2024 = df_2024_raw.rename(columns={'Latitude':'lat_2024','Longitude':'lon_2024','Clearance':'clearance_2024'})

    df_2022["clearance_2022"] = pd.to_numeric(df_2022["clearance_2022"], errors="coerce")
    df_2023["clearance_2023"] = pd.to_numeric(df_2023["clearance_2023"], errors="coerce")
    df_2024["clearance_2024"] = pd.to_numeric(df_2024["clearance_2024"], errors="coerce")

    df_2022 = df_2022.dropna(subset=['lat_2022','lon_2022'])
    df_2023 = df_2023.dropna(subset=['lat_2023','lon_2023'])
    df_2024 = df_2024.dropna(subset=['lat_2024','lon_2024'])

    # --- STEP 2 & 3a: GeoDataFrames & Spatial Join (15m) ---
    gdf_22 = gpd.GeoDataFrame(df_2022, geometry=gpd.points_from_xy(df_2022.lon_2022, df_2022.lat_2022), crs="EPSG:4326").to_crs("EPSG:3395")
    gdf_23 = gpd.GeoDataFrame(df_2023, geometry=gpd.points_from_xy(df_2023.lon_2023, df_2023.lat_2023), crs="EPSG:4326").to_crs("EPSG:3395")
    gdf_24 = gpd.GeoDataFrame(df_2024, geometry=gpd.points_from_xy(df_2024.lon_2024, df_2024.lat_2024), crs="EPSG:4326").to_crs("EPSG:3395")

    matched_22_23 = gpd.sjoin_nearest(gdf_22, gdf_23, how="inner", max_distance=15, distance_col="dist_22_23").drop(columns=["index_right"])
    final_matches = gpd.sjoin_nearest(matched_22_23, gdf_24, how="inner", max_distance=15, distance_col="dist_22_24")
    df = pd.DataFrame(final_matches.drop(columns="geometry"))

# --- STEP 3b: DYNAMIC Zone Logic (Dataset Agnostic) ---
    df["lat_2022"] = pd.to_numeric(df["lat_2022"], errors="coerce")
    df["lon_2022"] = pd.to_numeric(df["lon_2022"], errors="coerce")
    df = df.dropna(subset=["lat_2022", "lon_2022"])

    # 1. Dynamic North/South split (midpoint between highest and lowest latitude)
    lat_cutoff = (df["lat_2022"].min() + df["lat_2022"].max()) / 2

    # 2. Filter for points south of the cutoff
    south_mask = df["lat_2022"] < lat_cutoff

    # 3. Dynamic East/West split (median longitude of just the southern points)
    # Added a quick fallback just in case the southern half is completely empty
    if not df.loc[south_mask].empty:
        lon_cutoff = df.loc[south_mask, "lon_2022"].median()
    else:
        lon_cutoff = df["lon_2022"].median()

    # Apply the dynamic cutoffs to assign zones
    def zone_label(row):
        if row["lat_2022"] >= lat_cutoff:
            return "north"
        return "southeast" if row["lon_2022"] >= lon_cutoff else "southwest"
    
    df["zone"] = df.apply(zone_label, axis=1)

    # --- STEP 4 & 5: Growth, R-values, and k-NN Imputation ---
    df = df.dropna(subset=["clearance_2022", "clearance_2023", "clearance_2024"])
    df["growth_22_23"] = df["clearance_2022"] - df["clearance_2023"]
    df["growth_23_24"] = df["clearance_2023"] - df["clearance_2024"]
    df = df[(df["growth_22_23"] >= -4.0) & (df["growth_23_24"] >= -4.0)].copy()

    rain_22_23, rain_23_24 = 53.5, 64.0
    df["r1"] = df["growth_22_23"] / rain_22_23
    df["r2"] = df["growth_23_24"] / rain_23_24
    df["r_point"] = df[["r1","r2"]].mean(axis=1)

    gdf_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon_2022"], df["lat_2022"]), crs="EPSG:4326").to_crs("EPSG:3395")
    df["x_m"], df["y_m"] = gdf_df.geometry.x, gdf_df.geometry.y
    coords = df[["x_m", "y_m"]].values

    nn = NearestNeighbors(n_neighbors=6, algorithm="ball_tree").fit(coords)
    _, idxs = nn.kneighbors(coords)

    r = df["r_point"].astype(float).values
    r_new = r.copy()
    for i in range(len(df)):
        if np.isfinite(r[i]) and (r[i] < 0):
            neigh_r = r[idxs[i, 1:]]
            neigh_r = neigh_r[np.isfinite(neigh_r) & (neigh_r >= 0)]
            if len(neigh_r) > 0:
                r_new[i] = neigh_r.mean()

    df["r_point_nn5"] = r_new
    q_low, q_high = pd.Series(df["r_point_nn5"]).quantile([0.01, 0.99])
    df["r_point_nn5_clipped"] = df["r_point_nn5"].clip(q_low, q_high)

    return df

# ====================================================================
# FUNCTION 2: FAST SCENARIO MATH (Runs instantly when sliders move)
# ====================================================================
def predict_and_cluster(df, rain_forecast, w_loc, w_growth, danger_threshold):
    """Executes Steps 6 and 7 of Vegeman2_1.py based on GUI inputs."""
    
    # Predict Growth based on GUI Rain
    df["predicted_growth_24_25"] = df["r_point_nn5_clipped"] * rain_forecast
    df["predicted_clearance_2025"] = df["clearance_2024"] - df["predicted_growth_24_25"]

    # Assign GUI-based Danger Colors
    def get_color(c):
        if c < danger_threshold: return "red"
        if c < 6: return "orange"
        if c < 8: return "yellow"
        if c < 10: return "green"
        return "lightgray"
    
    df["risk_color"] = df["predicted_clearance_2025"].apply(get_color)

    def get_bucket(c):
        if c < danger_threshold: return f"<{danger_threshold} ft"
        if c < 6: return "4–6 ft"
        if c < 8: return "6–8 ft"
        if c < 10: return "8–10 ft"
        return "≥10 ft"
    
    df["risk_bucket"] = df["predicted_clearance_2025"].apply(get_bucket)

    # PER-ZONE Clustering based on GUI Location Weight
    n_clusters = 5
    df["cluster_location_growth"] = -1

    for zname, zdf in df.groupby("zone"):
        if len(zdf) < n_clusters: continue
        
        X = zdf[["lon_2022", "lat_2022", "predicted_growth_24_25"]].values
        X_scaled = StandardScaler().fit_transform(X)
        
        X_scaled[:, 0] *= w_loc
        X_scaled[:, 1] *= w_loc
        X_scaled[:, 2] *= w_growth
        
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_scaled)
        
        zone_offset = {"north": 0, "southwest": 100, "southeast": 200}.get(zname, 0)
        df.loc[zdf.index, "cluster_location_growth"] = labels + zone_offset

    return df