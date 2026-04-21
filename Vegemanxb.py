# =====================================================
# IMPORTS
# =====================================================

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
import contextily as ctx

# =============================================================
# 0. Convert .csv without Clearance to .csv with Clearance data
# =============================================================

INPUT_FILE = "capstone2022_ext2(in).csv"
OUTPUT_FILE = "capstone2022_output.csv"

U_COLS = [
    "encroachment.length.u0",
    "encroachment.length.u1",
    "encroachment.length.u2",
    "encroachment.length.u3",
    "encroachment.length.u4",
]

CLEARANCE_MAP = [2.0, 5.0, 7.0, 9.0, 11.0]

def compute_clearance(row):
    u0 = row["encroachment.length.u0"]

    # NA only appears in u0
    if pd.isna(u0) or str(u0).strip().upper() == "NA":
        return np.nan

    for i, col in enumerate(U_COLS):
        if float(row[col]) > 0:
            return CLEARANCE_MAP[i]

    return 0.0

df = pd.read_csv(INPUT_FILE)

df["Clearance"] = df.apply(compute_clearance, axis=1)

df.to_csv(OUTPUT_FILE, index=False)
print(f"Output written to {OUTPUT_FILE}")

print("\n=== Sanity: 2022 Clearance creation ===")
print("Rows:", len(df))
print("Clearance NA count:", df["Clearance"].isna().sum() if df["Clearance"].dtype != object else (df["Clearance"]=="NA").sum())
print("Clearance value counts (top):")
print(df["Clearance"].value_counts(dropna=False).head(10))

# =====================================================
# 1. LOAD RAW YEARLY DATA (original method)
# =====================================================
cols_to_use = ['Latitude', 'Longitude', 'Clearance']

df_2022 = pd.read_csv("capstone2022_output.csv", usecols=cols_to_use)
df_2023 = pd.read_csv("capstone2023_ext2(in).csv", usecols=cols_to_use)
df_2024 = pd.read_csv("capstone2024_ext2(in).csv", usecols=cols_to_use)

df_2022 = df_2022.rename(columns={'Latitude':'lat_2022','Longitude':'lon_2022','Clearance':'clearance_2022'})
df_2023 = df_2023.rename(columns={'Latitude':'lat_2023','Longitude':'lon_2023','Clearance':'clearance_2023'})
df_2024 = df_2024.rename(columns={'Latitude':'lat_2024','Longitude':'lon_2024','Clearance':'clearance_2024'})

df_2022["clearance_2022"] = pd.to_numeric(df_2022["clearance_2022"], errors="coerce")
df_2023["clearance_2023"] = pd.to_numeric(df_2023["clearance_2023"], errors="coerce")
df_2024["clearance_2024"] = pd.to_numeric(df_2024["clearance_2024"], errors="coerce")

print("\n=== Sanity: yearly clearance numeric conversion ===")
for name, d, c in [
    ("2022", df_2022, "clearance_2022"),
    ("2023", df_2023, "clearance_2023"),
    ("2024", df_2024, "clearance_2024"),
]:
    print(f"{name}: rows={len(d)}, missing_clearance={d[c].isna().sum()} ({d[c].isna().mean():.2%})")

# Drop invalid coords
df_2022 = df_2022.dropna(subset=['lat_2022','lon_2022'])
df_2023 = df_2023.dropna(subset=['lat_2023','lon_2023'])
df_2024 = df_2024.dropna(subset=['lat_2024','lon_2024'])

# =====================================================
# 2. CONVERT TO GEODATAFRAMES
# =====================================================
gdf_2022 = gpd.GeoDataFrame(df_2022, geometry=gpd.points_from_xy(df_2022.lon_2022, df_2022.lat_2022), crs="EPSG:4326")
gdf_2023 = gpd.GeoDataFrame(df_2023, geometry=gpd.points_from_xy(df_2023.lon_2023, df_2023.lat_2023), crs="EPSG:4326")
gdf_2024 = gpd.GeoDataFrame(df_2024, geometry=gpd.points_from_xy(df_2024.lon_2024, df_2024.lat_2024), crs="EPSG:4326")

# Reproject to EPSG:3395 for distance matching
gdf_2022p = gdf_2022.to_crs("EPSG:3395")
gdf_2023p = gdf_2023.to_crs("EPSG:3395")
gdf_2024p = gdf_2024.to_crs("EPSG:3395")

# =====================================================
# 3a. MATCHING LOGIC (IDENTICAL TO ORIGINAL SCRIPT)
# =====================================================
matched_22_23 = gpd.sjoin_nearest(
    gdf_2022p,
    gdf_2023p,
    how="inner",
    max_distance=15,
    distance_col="dist_22_23"
)
matched_22_23 = matched_22_23.drop(columns=["index_right"])

final_matches = gpd.sjoin_nearest(
    matched_22_23,
    gdf_2024p,
    how="inner",
    max_distance=15,
    distance_col="dist_22_24"
)

# Convert back to pandas
df = pd.DataFrame(final_matches.drop(columns="geometry"))

print("\n=== Sanity: spatial matching ===")
print("Matched rows (22→23→24):", len(df))
print("dist_22_23 (m):", df["dist_22_23"].describe())
print("dist_22_24 (m):", df["dist_22_24"].describe())

# =====================================================
# 3b. ADD ZONES (north / southwest / southeast)
# =====================================================

# Ensure numeric coords
df["lat_2022"] = pd.to_numeric(df["lat_2022"], errors="coerce")
df["lon_2022"] = pd.to_numeric(df["lon_2022"], errors="coerce")
df = df.dropna(subset=["lat_2022", "lon_2022"])

# Use below for future data sets
#lat_cutoff = (df["lat_2022"].min() + df["lat_2022"].max()) / 2
lat_cutoff = 28.0

# For south points, split east/west by median longitude
# south_mask = df["lat_2022"] < lat_cutoff

# Use below for future data sets
# lon_cutoff = df.loc[south_mask, "lon_2022"].median()
lon_cutoff = -81.0

def zone_label(row):
    if row["lat_2022"] >= lat_cutoff:
        return "north"
    return "southeast" if row["lon_2022"] >= lon_cutoff else "southwest"

df["zone"] = df.apply(zone_label, axis=1)

print(f"Zone cutoffs: lat_cutoff={lat_cutoff:.6f}, lon_cutoff={lon_cutoff:.6f}")
print(df["zone"].value_counts(dropna=False))

print("\n=== Sanity: zone ranges ===")
print(df.groupby("zone")[["lat_2022","lon_2022"]].agg(["min","max","mean","count"]))

# =====================================================
# 4. CLEAN + COMPUTE GROWTH
# =====================================================
df = df.dropna(subset=["clearance_2022", "clearance_2023", "clearance_2024"])

df["growth_22_23"] = df["clearance_2022"] - df["clearance_2023"]
df["growth_23_24"] = df["clearance_2023"] - df["clearance_2024"]

# Require at least 4 ft growth both years (positive growth = clearance drop)

# Save copy before filtering
df_before_growth_filter = df.copy()

# Keep-mask for model
growth_keep_mask = (df["growth_22_23"] >= -4.0) & (df["growth_23_24"] >= -4.0)

# Save filtered-out rows
df_filtered_out_growth = df_before_growth_filter.loc[~growth_keep_mask].copy()

# Keep remaining rows for the model
df = df_before_growth_filter.loc[growth_keep_mask].copy()

print("\n=== Sanity: after clearance/growth filtering ===")
print("Remaining rows:", len(df))
print("Filtered out from main model:", len(df_filtered_out_growth))
print(df[["growth_22_23","growth_23_24"]].describe())

df_filtered_out_growth.to_csv("filtered_out_main_model_points.csv", index=False)
print("Saved: filtered_out_main_model_points.csv")

# =====================================================
# 5. PER-POINT r VALUE
# =====================================================
rain_22_23 = 53.5
rain_23_24 = 64.0

df["r1"] = df["growth_22_23"] / rain_22_23
df["r2"] = df["growth_23_24"] / rain_23_24
df["r_point"] = df[["r1","r2"]].mean(axis=1)

q_low, q_high = df["r_point"].quantile([0.01,0.99])
df["r_point_clipped"] = df["r_point"].clip(q_low,q_high)

# =====================================================
# Replace negative r values with mean of 5 nearest neighbors
# =====================================================

from sklearn.neighbors import NearestNeighbors

# 1) Build projected x/y in meters from your already-projected 2022 GeoDataFrame
# We need the projected geometry for the SAME points in df.
# Easiest: rebuild a GeoDataFrame from df's 2022 coords and project it.

gdf_df = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon_2022"], df["lat_2022"]),
    crs="EPSG:4326"
).to_crs("EPSG:3395")

df["x_m"] = gdf_df.geometry.x
df["y_m"] = gdf_df.geometry.y

# 2) Fit nearest neighbors model
coords = df[["x_m", "y_m"]].values
nn = NearestNeighbors(n_neighbors=6, algorithm="ball_tree")
nn.fit(coords)

dists, idxs = nn.kneighbors(coords)

# 3) Replace negative r_point with mean of 5 nearest neighbors (excluding self)
r = df["r_point"].astype(float).values
r_new = r.copy()

for i in range(len(df)):
    if np.isfinite(r[i]) and (r[i] < 0):
        neigh_idx = idxs[i, 1:]
        neigh_r = r[neigh_idx]

        # only use non-negative neighbor r's
        neigh_r = neigh_r[np.isfinite(neigh_r) & (neigh_r >= 0)]

        # if we have at least 1 valid neighbor, replace; else leave as-is
        if len(neigh_r) > 0:
            r_new[i] = neigh_r.mean()

df["r_point_nn5"] = r_new

# Optional: clip the smoothed r to avoid crazies
q_low, q_high = pd.Series(df["r_point_nn5"]).quantile([0.01, 0.99])
df["r_point_nn5_clipped"] = df["r_point_nn5"].clip(q_low, q_high)

print("Replaced negative r_point values using nearest-neighbor mean (k=5).")
print("New columns: r_point_nn5, r_point_nn5_clipped")


# =====================================================
# 6. PREDICT 2024→2025
# =====================================================
rain_24_25 = 48.7  # adjust scenario

df["predicted_growth_24_25"] = df["r_point_nn5_clipped"] * rain_24_25
df["predicted_clearance_2025"] = df["clearance_2024"] - df["predicted_growth_24_25"]

# Risk color & bucket
def color(c):
    if c < 4: return "red"
    if c < 6: return "orange"
    if c < 8: return "yellow"
    if c < 10: return "green"
    return "lightgray"

df["risk_color"] = df["predicted_clearance_2025"].apply(color)

def bucket(c):
    if c < 4: return "<4 ft"
    if c < 6: return "4–6 ft"
    if c < 8: return "6–8 ft"
    if c < 10: return "8–10 ft"
    return "≥10 ft"

df["risk_bucket"] = df["predicted_clearance_2025"].apply(bucket)

# =====================================================
# 7. CLUSTERING (PER ZONE)
# =====================================================
n_clusters = 5  # per-zone clusters

df["cluster_location"] = -1

for zname, zdf in df.groupby("zone"):
    if len(zdf) < n_clusters:
        continue

    Xloc = StandardScaler().fit_transform(zdf[["lon_2022","lat_2022"]])
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(Xloc)

    # Make cluster IDs unique across zones (so north cluster 0 != south cluster 0)
    zone_offset = {"north": 0, "southwest": 100, "southeast": 200}.get(zname, 0)
    df.loc[zdf.index, "cluster_location"] = labels + zone_offset

print("Clustered per zone into df['cluster_location']")

# =====================================================
# 8. SAVE FINAL OUTPUT CSV
# =====================================================
df.to_csv("per_point_growth_predictions_with_clusters.csv", index=False)
print("SAVED: per_point_growth_predictions_with_clusters.csv")

# =====================================================
# 9. Maps
# =====================================================

def plot_zone_basemap(
    zdf: pd.DataFrame,
    title: str,
    color_col: str = None,
    cmap: str = None,
    size: int = 12,
    alpha: float = 0.9,
    add_colorbar: bool = False,
    cbar_label: str = "",
    legend_handles=None,
):
    # Make GeoDataFrame and project to Web Mercator for basemap tiles
    g = gpd.GeoDataFrame(
        zdf.copy(),
        geometry=gpd.points_from_xy(zdf["lon_2022"], zdf["lat_2022"]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter
    if cmap is None:
        sc = ax.scatter(g.geometry.x, g.geometry.y, c=g[color_col], s=size, alpha=alpha)
    else:
        sc = ax.scatter(g.geometry.x, g.geometry.y, c=g[color_col], s=size, alpha=alpha, cmap=cmap)

    # Basemap (Esri)
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

    # Optional legend (for categorical colors)
    if legend_handles is not None:
        ax.legend(handles=legend_handles, title="Legend", loc="lower left")

    # Optional colorbar (for continuous)
    if add_colorbar:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)

    ax.set_axis_off()
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# =====================================================
# 9a. Main model filtered-out points PER ZONE
# =====================================================

for zname in sorted(df_before_growth_filter["zone"].dropna().unique()):
    kept_z = df[df["zone"] == zname].copy()
    filt_z = df_filtered_out_growth[df_filtered_out_growth["zone"] == zname].copy()

    g_kept = gpd.GeoDataFrame(
        kept_z,
        geometry=gpd.points_from_xy(kept_z["lon_2022"], kept_z["lat_2022"]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    g_filt = gpd.GeoDataFrame(
        filt_z,
        geometry=gpd.points_from_xy(filt_z["lon_2022"], filt_z["lat_2022"]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 8))

    if len(g_kept) > 0:
        ax.scatter(
            g_kept.geometry.x, g_kept.geometry.y,
            c="gray", s=8, alpha=0.35, label="Kept"
        )

    if len(g_filt) > 0:
        ax.scatter(
            g_filt.geometry.x, g_filt.geometry.y,
            c="red", s=18, alpha=0.95, label="Filtered out"
        )

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

    ax.legend(loc="lower left")
    ax.set_axis_off()
    ax.set_title(f"Main Model: Kept vs Filtered-Out Points ({zname})")
    plt.tight_layout()
    plt.show()

# =====================================================
# 9b. Predicted clearance risk map PER ZONE
# =====================================================

legend_patches = [
    mpatches.Patch(color="red",       label="< 4 ft"),
    mpatches.Patch(color="orange",    label="4–6 ft"),
    mpatches.Patch(color="yellow",    label="6–8 ft"),
    mpatches.Patch(color="green",     label="8–10 ft"),
    mpatches.Patch(color="lightgray", label="≥ 10 ft"),
]

for zname, zdf in df.groupby("zone"):
    plot_zone_basemap(
        zdf,
        title=f"Predicted Clearance to Wire (2025) – Risk Map ({zname})",
        color_col="risk_color",
        cmap=None,
        size=12,
        alpha=0.9,
        legend_handles=legend_patches,
    )

# =====================================================
# 9c. Predicted growth heatmap PER ZONE
# =====================================================

for zname, zdf in df.groupby("zone"):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        zdf["lon_2022"],
        zdf["lat_2022"],
        c=zdf["predicted_growth_24_25"],
        s=10,
        alpha=0.9,
        cmap="viridis",
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Predicted Growth 2024→2025 (ft)")

    plt.title(f"Predicted Vegetation Growth 2024→2025 ({zname})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# =====================================================
# 10. TABLES
# =====================================================

# TABLE A: Summary by risk bucket (for risk map using zones)
risk_summary_by_zone = (
    df.groupby(["zone", "risk_bucket"])
      .agg(
          total_points=("predicted_clearance_2025", "size"),
          mean_clearance_ft=("predicted_clearance_2025", "mean"),
          min_clearance_ft=("predicted_clearance_2025", "min"),
          max_clearance_ft=("predicted_clearance_2025", "max"),
      )
      .reset_index()
)

# percent within zone
zone_totals = df.groupby("zone").size().rename("zone_total").reset_index()
risk_summary_by_zone = risk_summary_by_zone.merge(zone_totals, on="zone", how="left")
risk_summary_by_zone["percent_of_zone"] = 100.0 * risk_summary_by_zone["total_points"] / risk_summary_by_zone["zone_total"]

print(risk_summary_by_zone.to_string(index=False))
risk_summary_by_zone.to_csv("table_risk_by_zone_and_bucket.csv", index=False)
print("Saved: table_risk_by_zone_and_bucket.csv")

# =====================================================
# TABLE C: Summary by location-based cluster (spatial zones)
# =====================================================

location_cluster_summary = (
    df.groupby("cluster_location")
      .agg(
          total_points=("predicted_clearance_2025", "size"),
          centroid_lon=("lon_2022", "mean"),
          centroid_lat=("lat_2022", "mean"),
          mean_r_point=("r_point", "mean"),
          mean_predicted_growth_24_25=("predicted_growth_24_25", "mean"),
          mean_predicted_clearance_2025_ft=("predicted_clearance_2025", "mean"),
      )
      .reset_index()
)

print("\n=== TABLE C: Clusters by location (spatial zones) ===")
print(location_cluster_summary.to_string(index=False))

location_cluster_summary.to_csv(
    "table_clusters_by_location.csv", index=False
)

print("Saved: table_clusters_by_location.csv")


# TABLE D: Location clusters vs 2025 risk buckets (% of points)
counts = (
    df.groupby(["cluster_location", "risk_bucket"])
      .size()
      .reset_index(name="count")
)

totals = (
    df.groupby("cluster_location")["predicted_clearance_2025"]
      .size()
      .reset_index(name="total_points")
)

loc_risk = counts.merge(totals, on="cluster_location", how="left")
loc_risk["percent_of_cluster"] = 100.0 * loc_risk["count"] / loc_risk["total_points"]

loc_risk_pivot = loc_risk.pivot(
    index="cluster_location",
    columns="risk_bucket",
    values="percent_of_cluster",
).fillna(0.0).reset_index()

print("\n=== TABLE D: Location clusters vs 2025 risk buckets (% of points) ===")
print(loc_risk_pivot.to_string(index=False))

loc_risk_pivot.to_csv("table_location_clusters_risk_buckets.csv", index=False)
print("Saved: table_location_clusters_risk_buckets.csv")

# =====================================================
# CLUSTERING: Location + Predicted Growth PER ZONE (adjustable weights)
# =====================================================

n_clusters = 5
w_loc = 2.25
w_growth = 1.0

df["cluster_location_growth"] = -1

for zname, zdf in df.groupby("zone"):
    if len(zdf) < n_clusters:
        continue

    X = zdf[["lon_2022", "lat_2022", "predicted_growth_24_25"]].values
    X_scaled = StandardScaler().fit_transform(X)

    X_scaled[:, 0] *= w_loc
    X_scaled[:, 1] *= w_loc
    X_scaled[:, 2] *= w_growth

    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_scaled)

    zone_offset = {"north": 0, "southwest": 100, "southeast": 200}.get(zname, 0)
    df.loc[zdf.index, "cluster_location_growth"] = labels + zone_offset

print("Created df['cluster_location_growth'] per zone.")

# =================================================================
# MAP: One color per cluster (location + predicted growth) PER ZONE
# =================================================================

for zname, zdf in df.groupby("zone"):
    plt.figure(figsize=(10, 8))

    cluster_ids = sorted(zdf["cluster_location_growth"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(cluster_ids))))

    for cid, col in zip(cluster_ids, colors):
        mask = zdf["cluster_location_growth"] == cid
        plt.scatter(
            zdf.loc[mask, "lon_2022"],
            zdf.loc[mask, "lat_2022"],
            c=[col],
            s=10,
            alpha=0.9,
            label=f"Cluster {cid}"
        )

    plt.title(f"Clusters by Location + Predicted Growth ({zname})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(markerscale=2, fontsize=9, title="Loc+Growth cluster")
    plt.tight_layout()
    plt.show()

# =====================================================
# TABLE: Risk bucket percentages per Loc+Growth cluster
# =====================================================

counts = (
    df.groupby(["cluster_location_growth", "risk_bucket"])
      .size()
      .reset_index(name="count")
)

totals = (
    df.groupby("cluster_location_growth")
      .size()
      .reset_index(name="total_points")
)

risk_by_cluster = counts.merge(totals, on="cluster_location_growth", how="left")
risk_by_cluster["percent_of_cluster"] = (
    100.0 * risk_by_cluster["count"] / risk_by_cluster["total_points"]
)

risk_by_cluster_pivot = (
    risk_by_cluster
      .pivot(
          index="cluster_location_growth",
          columns="risk_bucket",
          values="percent_of_cluster"
      )
      .fillna(0.0)
      .reset_index()
)

print("\n=== TABLE: Risk Distribution by Location + Predicted Growth Cluster ===")
print(risk_by_cluster_pivot.to_string(index=False))

risk_by_cluster_pivot.to_csv(
    "table_loc_growth_clusters_risk_buckets.csv",
    index=False
)

print("Saved: table_loc_growth_clusters_risk_buckets.csv")


# =====================================================
# 11. VALIDATION AGAINST REAL 2025 LIDAR
# =====================================================

# ---- A) Load 2025 real data ----
cols_2025 = ["Latitude", "Longitude", "Clearance"]
df_2025 = pd.read_csv("capstone2025.csv", usecols=cols_2025)

df_2025 = df_2025.rename(columns={
    "Latitude": "lat_2025",
    "Longitude": "lon_2025",
    "Clearance": "clearance_2025_actual"
 })
df_2025 = df_2025.dropna(subset=["lat_2025", "lon_2025", "clearance_2025_actual"])

# Ensure numeric clearance
df_2025["clearance_2025_actual"] = pd.to_numeric(df_2025["clearance_2025_actual"], errors="coerce")
df_2025 = df_2025.dropna(subset=["clearance_2025_actual"])

# ---- B) Build GeoDataFrames for matching (same logic as your original) ----
# We'll match model points (use 2024 coords) → nearest 2025 point

df_model = df.copy()

# Make sure model has 2024 lat/lon
for c in ["lat_2024", "lon_2024"]:
  if c not in df_model.columns:
      raise ValueError(f"Missing {c} in model dataframe. Make sure you kept 2024 coords.")

gdf_model = gpd.GeoDataFrame(
    df_model,
    geometry=gpd.points_from_xy(df_model["lon_2024"], df_model["lat_2024"]),
    crs="EPSG:4326"
)

gdf_2025 = gpd.GeoDataFrame(
    df_2025,
    geometry=gpd.points_from_xy(df_2025["lon_2025"], df_2025["lat_2025"]),
    crs="EPSG:4326"
)

# Reproject to meters
gdf_model_m = gdf_model.to_crs("EPSG:3395")
gdf_2025_m  = gdf_2025.to_crs("EPSG:3395")

# ---- C) Nearest neighbor join (15 m) ----

# Clean up any leftover sjoin index columns to avoid GeoPandas conflicts
def drop_join_index_cols(gdf):
    drop_cols = []
    for c in gdf.columns:
        if c in ["index_left", "index_right"]:
            drop_cols.append(c)
        if c.startswith("index_") and c not in ["index", "index_x", "index_y"]:
            drop_cols.append(c)
    if drop_cols:
        gdf = gdf.drop(columns=list(set(drop_cols)), errors="ignore")
    return gdf

gdf_model_m = drop_join_index_cols(gdf_model_m)
gdf_2025_m  = drop_join_index_cols(gdf_2025_m)

validation_join = gpd.sjoin_nearest(
    gdf_model_m,
    gdf_2025_m,
    how="inner",
    max_distance=15,
    distance_col="dist_24_to_25_m",
    lsuffix="model",
    rsuffix="y2025"
)

# Drop the right-side index column created by this join
if "index_y2025" in validation_join.columns:
    validation_join = validation_join.drop(columns=["index_y2025"])
elif "index_right" in validation_join.columns:
    validation_join = validation_join.drop(columns=["index_right"])

# Convert back to pandas
val = pd.DataFrame(validation_join.drop(columns="geometry")).copy()

if "zone" not in val.columns:
    raise ValueError("Zone column missing in validation results. Ensure df has df['zone'] before validation join.")

# ---- D) Compute errors ----
# predicted_clearance_2025 comes from your model
if "predicted_clearance_2025" not in val.columns:
    raise ValueError("Missing predicted_clearance_2025. Make sure prediction step ran before validation.")

val["error_clearance_ft"] = val["predicted_clearance_2025"] - val["clearance_2025_actual"]
val["abs_error_clearance_ft"] = val["error_clearance_ft"].abs()

# Optional: actual growth from 2024→2025 if you have clearance_2024
if "clearance_2024" in val.columns:
    val["actual_growth_24_25"] = val["clearance_2024"] - val["clearance_2025_actual"]
    val["error_growth_ft"] = val["predicted_growth_24_25"] - val["actual_growth_24_25"]
    val["abs_error_growth_ft"] = val["error_growth_ft"].abs()

# =====================================================
# OPTIONAL FILTER: Remove locations with <= -4 ft actual growth (2024→2025)
# (i.e., clearance increased by 4+ ft; often trimming/intervention)
# =====================================================

# Make sure actual growth exists
if "actual_growth_24_25" not in val.columns:
    val["actual_growth_24_25"] = val["clearance_2024"] - val["clearance_2025_actual"]

# Save copy before validation trim/intervention filter
val_before_trim_filter = val.copy()

# Keep-mask for validation
trim_keep_mask = val["actual_growth_24_25"] > -3.0

# Save filtered-out validation rows
val_filtered_out_trim = val_before_trim_filter.loc[~trim_keep_mask].copy()

# Keep remaining rows for validation metrics
val = val_before_trim_filter.loc[trim_keep_mask].copy()

print(f"\nFiltered out {len(val_filtered_out_trim)} points with actual_growth_24_25 <= -3.0 ft (likely trimming).")
print(f"Remaining validation points: {len(val)}")

val_filtered_out_trim.to_csv("filtered_out_validation_trim_points.csv", index=False)
print("Saved: filtered_out_validation_trim_points.csv")

# ---- E) Summary metrics ----
mae = val["abs_error_clearance_ft"].mean()
rmse = np.sqrt((val["error_clearance_ft"]**2).mean())
bias = val["error_clearance_ft"].mean()

# R^2 for clearance prediction (if there is variance)
y_true = val["clearance_2025_actual"].values
y_pred = val["predicted_clearance_2025"].values
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
std_err_clearance = val["error_clearance_ft"].std()
std_abs_err_clearance = val["abs_error_clearance_ft"].std()

# Optional: growth error std dev (if you want it too)
std_err_growth = val["error_growth_ft"].std() if "error_growth_ft" in val.columns else None

print("\n==============================")
print("  VALIDATION: Predicted vs Actual 2025 Clearance")
print("==============================")
print(f"Matched points: {len(val)}")
print(f"MAE  (ft): {mae:.4f}")
print(f"RMSE (ft): {rmse:.4f}")
print(f"Bias (ft): {bias:.4f}   (positive = overpredict clearance)")
print(f"Std Err (ft): {std_err_clearance:.4f}   (std dev of [Pred-Actual])")
print(f"Std |Err| (ft): {std_abs_err_clearance:.4f} (std dev of absolute error)")
print(f"R²        : {r2:.4f}")

if std_err_growth is not None:
    print(f"Std Growth Err (ft): {std_err_growth:.4f} (std dev of growth error)")

print("==============================\n")


# ---- F) Save validation results per-point ----
val.to_csv("validation_predicted_vs_actual_2025.csv", index=False)
print("Saved: validation_predicted_vs_actual_2025.csv")

# ---- G) Plots ----

# 1) Scatter: predicted vs actual clearance
plt.figure(figsize=(7, 6))
plt.scatter(val["clearance_2025_actual"], val["predicted_clearance_2025"], s=10, alpha=0.6)
minv = min(val["clearance_2025_actual"].min(), val["predicted_clearance_2025"].min())
maxv = max(val["clearance_2025_actual"].max(), val["predicted_clearance_2025"].max())
plt.plot([minv, maxv], [minv, maxv], linestyle="--")  # 1:1 line
plt.xlabel("Actual Clearance 2025 (ft)")
plt.ylabel("Predicted Clearance 2025 (ft)")
plt.title("Predicted vs Actual Clearance (2025)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# 2) Histogram of clearance error
plt.figure(figsize=(7, 5))
plt.hist(val["error_clearance_ft"].dropna(), bins=40, alpha=0.8)
plt.xlabel("Prediction Error (ft)  [Pred - Actual]")
plt.ylabel("Count")
plt.title("Clearance Prediction Error Distribution (2025)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# 3) Map of absolute error (heat-style) PER ZONE
for zname, vz in val.groupby("zone"):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        vz["lon_2024"], vz["lat_2024"],
        c=vz["abs_error_clearance_ft"],
        s=10,
        alpha=0.85,
        cmap="magma"
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Absolute Error in Clearance (ft)")
    plt.title(f"Spatial Error Map: |Predicted - Actual| Clearance (2025) ({zname})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

# =====================================================
# Validation points filtered out by trim/intervention rule PER ZONE
# =====================================================

for zname in sorted(val_before_trim_filter["zone"].dropna().unique()):
    kept_z = val[val["zone"] == zname].copy()
    filt_z = val_filtered_out_trim[val_filtered_out_trim["zone"] == zname].copy()

    g_kept = gpd.GeoDataFrame(
        kept_z,
        geometry=gpd.points_from_xy(kept_z["lon_2024"], kept_z["lat_2024"]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    g_filt = gpd.GeoDataFrame(
        filt_z,
        geometry=gpd.points_from_xy(filt_z["lon_2024"], filt_z["lat_2024"]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 8))

    if len(g_kept) > 0:
        ax.scatter(
            g_kept.geometry.x, g_kept.geometry.y,
            c="gray", s=8, alpha=0.35, label="Kept for validation"
        )

    if len(g_filt) > 0:
        ax.scatter(
            g_filt.geometry.x, g_filt.geometry.y,
            c="cyan", s=20, alpha=0.95, label="Filtered out (likely trimming)"
        )

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

    ax.legend(loc="lower left")
    ax.set_axis_off()
    ax.set_title(f"Validation: Kept vs Trim-Filtered Points ({zname})")
    plt.tight_layout()
    plt.show()

# Optional: growth validation plot
if "actual_growth_24_25" in val.columns:
    plt.figure(figsize=(7, 6))
    plt.scatter(val["actual_growth_24_25"], val["predicted_growth_24_25"], s=10, alpha=0.6)
    minv = min(val["actual_growth_24_25"].min(), val["predicted_growth_24_25"].min())
    maxv = max(val["actual_growth_24_25"].max(), val["predicted_growth_24_25"].max())
    plt.plot([minv, maxv], [minv, maxv], linestyle="--")
    plt.xlabel("Actual Growth 2024→2025 (ft)")
    plt.ylabel("Predicted Growth 2024→2025 (ft)")
    plt.title("Predicted vs Actual Growth (2024→2025)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

# =====================================================
# ACCURACY BY GROWTH ERROR BANDS
# =====================================================

# Make sure required columns exist
required = ["predicted_growth_24_25", "actual_growth_24_25"]
missing = [c for c in required if c not in val.columns]
if missing:
    raise ValueError(f"Missing required columns for growth accuracy: {missing}")

# Absolute growth error
val["abs_growth_error_ft"] = (
    val["predicted_growth_24_25"] - val["actual_growth_24_25"]
).abs()

# Define tolerance bands (ft)
bands = [0.5, 1.0, 2.0, 3.0, 4.0]

rows = []
n = len(val)

for b in bands:
    pct = 100.0 * (val["abs_growth_error_ft"] <= b).sum() / n
    rows.append({
        "tolerance_ft": f"±{b}",
        "points_within": (val["abs_growth_error_ft"] <= b).sum(),
        "percent_within": pct
    })

accuracy_table = pd.DataFrame(rows)

print("\n=== Growth Prediction Accuracy by Tolerance ===")
print(accuracy_table.to_string(index=False))

accuracy_table.to_csv(
    "table_growth_accuracy_by_tolerance.csv",
    index=False
)

print("Saved: table_growth_accuracy_by_tolerance.csv")

plt.figure(figsize=(7, 5))
plt.plot(
    accuracy_table["tolerance_ft"],
    accuracy_table["percent_within"],
    marker="o"
)
plt.ylabel("Percent of Points Within Tolerance (%)")
plt.xlabel("Growth Error Tolerance (ft)")
plt.title("Growth Prediction Accuracy vs Tolerance")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================================
# DANGER ZONE (4 ft) MISSES: Actual <4 but Predicted >=4
# =====================================================

threshold = 4.0

# Ensure columns exist
if "predicted_clearance_2025" not in val.columns or "clearance_2025_actual" not in val.columns:
    raise ValueError("Missing predicted_clearance_2025 or clearance_2025_actual in val.")

pred_danger = val["predicted_clearance_2025"] < threshold
actual_danger = val["clearance_2025_actual"] < threshold

TP = int((pred_danger & actual_danger).sum())
FP = int((pred_danger & ~actual_danger).sum())
FN = int((~pred_danger & actual_danger).sum())  # MISSED DANGER
TN = int((~pred_danger & ~actual_danger).sum())

total = len(val)

# Rates
recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan          # "catch rate"
precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan       # how many flagged were real
miss_rate = FN / (TP + FN) if (TP + FN) > 0 else np.nan       # missed share of all actual danger
false_alarm_rate = FP / (FP + TN) if (FP + TN) > 0 else np.nan

print("\n==============================")
print("  DANGER ZONE PERFORMANCE (Threshold: 4 ft)")
print("==============================")
print(f"Total matched validation points: {total}")
print(f"TP (caught danger)        : {TP}")
print(f"FP (false alarms)         : {FP}")
print(f"FN (MISSED danger)        : {FN}")
print(f"TN (correct safe)         : {TN}")
print("------------------------------")
print(f"Recall / Catch rate       : {recall:.4f}")
print(f"Precision                 : {precision:.4f}")
print(f"Miss rate (FN share)      : {miss_rate:.4f}")
print(f"False alarm rate          : {false_alarm_rate:.4f}")
print("==============================\n")

# Missed danger points per zone
for zname, vz in val.groupby("zone"):
    missed_z = vz[(vz["clearance_2025_actual"] < threshold) & (vz["predicted_clearance_2025"] >= threshold)].copy()

    plt.figure(figsize=(10, 8))
    plt.scatter(vz["lon_2024"], vz["lat_2024"], s=6, alpha=0.25, label="All validation points")
    plt.scatter(missed_z["lon_2024"], missed_z["lat_2024"], s=18, alpha=0.9, label="MISSED danger (<4ft actual)")
    plt.title(f"Missed Danger Points: Actual <4ft but Predicted >=4ft ({zname})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"{zname}: missed danger points = {len(missed_z)} / {len(vz)}")

print("\n================ FINAL SANITY CHECK ================")

print("Total modeled points (after all filters):", len(df))
print("Total validation points (post-filter):", len(val))

print("\nPredicted clearance 2025 summary (ft):")
print(df["predicted_clearance_2025"].describe())

print("\nValidation clearance error summary (ft):")
print(val["error_clearance_ft"].describe())

print("\nZone counts:")
print(df["zone"].value_counts())

print("\nRisk bucket distribution:")
print(df["risk_bucket"].value_counts())

print("====================================================")

print("\nSummary metrics:")
print(f"MAE (ft):  {mae:.3f}")
print(f"RMSE (ft): {rmse:.3f}")
print(f"Bias (ft): {bias:.3f}")
print(f"R²:        {r2:.3f}")
print(f"Recall (<4 ft):     {recall:.3f}")
print(f"False alarm rate:   {false_alarm_rate:.3f}")
print(f"Miss rate (<4 ft):  {miss_rate:.3f}")