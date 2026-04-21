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
from sklearn.neighbors import NearestNeighbors

!pip install contextily

# =====================================================
# 1. LOAD RAW YEARLY DATA (original method)
# =====================================================
cols_to_use_22_23 = ["Latitude", "Longitude", "Clearance"]
cols_to_use_2024  = ["Latitude", "Longitude", "Clearance", "encroachment.volume.u0", "encroachment.volume.u1"]

df_2022 = pd.read_csv("capstone2022.csv", usecols=cols_to_use_22_23)
df_2023 = pd.read_csv("capstone2023.csv", usecols=cols_to_use_22_23)
df_2024 = pd.read_csv("capstone2024.csv", usecols=cols_to_use_2024)

df_2022 = df_2022.rename(columns={'Latitude':'lat_2022','Longitude':'lon_2022','Clearance':'clearance_2022'})
df_2023 = df_2023.rename(columns={'Latitude':'lat_2023','Longitude':'lon_2023','Clearance':'clearance_2023'})
df_2024 = df_2024.rename(columns={
    "Latitude":"lat_2024","Longitude":"lon_2024","Clearance":"clearance_2024",
    "encroachment.volume.u0":"u0_2024","encroachment.volume.u1":"u1_2024"
})

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
# 3. MATCHING LOGIC (IDENTICAL TO ORIGINAL SCRIPT)
# =====================================================
matched_22_23 = gpd.sjoin_nearest(
    gdf_2022p,
    gdf_2023p,
    how="inner",
    max_distance=10,
    distance_col="dist_22_23"
)
matched_22_23 = matched_22_23.drop(columns=["index_right"])

final_matches = gpd.sjoin_nearest(
    matched_22_23,
    gdf_2024p,
    how="inner",
    max_distance=10,
    distance_col="dist_22_24"
)

# =====================================================
# SAG/BREAK FLAGS from 2024 volume columns (run on final_matches)
# Danger1: clearance_2024 > 4 ft AND u0_2024 volume > 0
# Danger2: clearance_2024 > 8 ft AND u1_2024 volume > 0
# =====================================================

final_matches["clearance_2024"] = pd.to_numeric(final_matches["clearance_2024"], errors="coerce")
final_matches["u0_2024"] = pd.to_numeric(final_matches["u0_2024"], errors="coerce").fillna(0)
final_matches["u1_2024"] = pd.to_numeric(final_matches["u1_2024"], errors="coerce").fillna(0)

final_matches["danger1_sagbreak"] = (final_matches["clearance_2024"] > 4.0) & (final_matches["u0_2024"] > 0)
final_matches["danger2_sagbreak"] = (final_matches["clearance_2024"] > 8.0) & (final_matches["u1_2024"] > 0)

final_matches["sagbreak_flag"] = np.select(
    [
        final_matches["danger1_sagbreak"] & final_matches["danger2_sagbreak"],
        final_matches["danger2_sagbreak"],
        final_matches["danger1_sagbreak"]
    ],
    ["danger1+danger2", "danger2", "danger1"],
    default="none"
)

print("Added sag/break flags to final_matches")
print(final_matches["sagbreak_flag"].value_counts(dropna=False))
print("Danger1 count:", int(final_matches["danger1_sagbreak"].sum()))
print("Danger2 count:", int(final_matches["danger2_sagbreak"].sum()))

# Convert back to pandas
df = pd.DataFrame(final_matches.drop(columns="geometry"))

print("Danger1 count:", int(df["danger1_sagbreak"].sum()))
print("Danger2 count:", int(df["danger2_sagbreak"].sum()))

# =====================================================
# 4. CLEAN + COMPUTE GROWTH
# =====================================================
df["growth_22_23"] = df["clearance_2022"] - df["clearance_2023"]
df["growth_23_24"] = df["clearance_2023"] - df["clearance_2024"]

# Require at least 4 ft growth both years (positive growth = clearance drop)
# df = df[(df["growth_22_23"] >= -4.0) & (df["growth_23_24"] >= -4.0)].copy()

df["avg_growth"] = 0.5 * (df["growth_22_23"] + df["growth_23_24"])

# =====================================================
# 5. PER-POINT r VALUE
# =====================================================
rain_22_23 = 53.5
rain_23_24 = 64.0

df["r1"] = df["growth_22_23"] / rain_22_23
df["r2"] = df["growth_23_24"] / rain_23_24

# Weighted toward recent growth
df["r_point"] = 0.3 * df["r1"] + 0.7 * df["r2"]

# Clip for stability
#q_low, q_high = df["r_point"].quantile([0.01, 0.99])
#df["r_point_clipped"] = df["r_point"].clip(q_low, q_high)

# =====================================================
# 6. PREDICT 2024→2025
# =====================================================
rain_24_25 = 48.7  # adjust scenario

# Base prediction from weighted r
df["predicted_growth_24_25"] = df["r_point_clipped"] * rain_24_25

# ---------------------------------
# Override for strong recent growth
# ---------------------------------
high_growth_mask = df["growth_23_24"] > 2.5
df.loc[high_growth_mask, "predicted_growth_24_25"] = (
    df.loc[high_growth_mask, "growth_23_24"] * 1.05
)

# ---------------------------------
# Boost near wire
# ---------------------------------
df["clearance_boost"] = np.where(df["clearance_2024"] < 6.0, 1.15, 1.0)
df["predicted_growth_24_25"] = df["predicted_growth_24_25"] * df["clearance_boost"]

# ---------------------------------
# Boost for vegetation density / sag-break volume
# ---------------------------------
df["density_boost"] = (
    1
    + 0.10 * (df["u0_2024"] > 0).astype(int)
    + 0.15 * (df["u1_2024"] > 0).astype(int)
)
df["predicted_growth_24_25"] = df["predicted_growth_24_25"] * df["density_boost"]

# Optional: clip predicted growth so insane outliers don't blow up the model
g_low, g_high = df["predicted_growth_24_25"].quantile([0.01, 0.99])
df["predicted_growth_24_25"] = df["predicted_growth_24_25"].clip(g_low, g_high)

# Final predicted clearance
df["predicted_clearance_2025"] = df["clearance_2024"] - df["predicted_growth_24_25"]

# Risk color & bucket
def color(c):
    if c < 4:
        return "red"
    if c < 6:
        return "orange"
    if c < 8:
        return "yellow"
    if c < 10:
        return "green"
    return "lightgray"

df["risk_color"] = df["predicted_clearance_2025"].apply(color)

def bucket(c):
    if c < 4:
        return "<4 ft"
    if c < 6:
        return "4–6 ft"
    if c < 8:
        return "6–8 ft"
    if c < 10:
        return "8–10 ft"
    return "≥10 ft"

df["risk_bucket"] = df["predicted_clearance_2025"].apply(bucket)

# =====================================================
# 7. CLUSTERING
# =====================================================
n_clusters = 10

Xloc = StandardScaler().fit_transform(df[["lon_2022","lat_2022"]])
df["cluster_location"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(Xloc)

# =====================================================
# 8. SAVE FINAL OUTPUT CSV
# =====================================================
df.to_csv("per_point_growth_predictions_with_clusters.csv", index=False)
print("SAVED: per_point_growth_predictions_with_clusters.csv")

# =====================================================
# 9. MAPS
# =====================================================

# =====================================================
# Predicted clearance risk map WITH ArcGIS basemap + legend
# =====================================================

import contextily as ctx
import matplotlib.patches as mpatches

gdf_plot = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon_2022"], df["lat_2022"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(
    gdf_plot.geometry.x,
    gdf_plot.geometry.y,
    c=gdf_plot["risk_color"],
    s=12,
    alpha=0.9
)

ctx.add_basemap(
    ax,
    source=ctx.providers.Esri.WorldImagery,
    zoom=16
)

legend_patches = [
    mpatches.Patch(color="red",       label="< 4 ft"),
    mpatches.Patch(color="orange",    label="4–6 ft"),
    mpatches.Patch(color="yellow",    label="6–8 ft"),
    mpatches.Patch(color="green",     label="8–10 ft"),
    mpatches.Patch(color="lightgray", label="≥ 10 ft"),
]

ax.legend(
    handles=legend_patches,
    title="Predicted Clearance 2025",
    loc="lower left"
)

ax.set_axis_off()
plt.title("Predicted Clearance to Wire (2025) – Risk Map")
plt.tight_layout()
plt.show()

# 9b. Predicted growth heatmap (continuous colors)
plt.figure(figsize=(10, 8))

sc = plt.scatter(
    df["lon_2022"],
    df["lat_2022"],
    c=df["predicted_growth_24_25"],
    s=10,
    alpha=0.9,
    cmap="viridis",
)

cbar = plt.colorbar(sc)
cbar.set_label("Predicted Growth 2024→2025 (ft)")

plt.title("Predicted Vegetation Growth 2024→2025")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# 9d. Cluster by location (one color per cluster)
plt.figure(figsize=(10, 8))

clusters_loc = sorted(df["cluster_location"].unique())
colors_loc = plt.cm.tab10(np.linspace(0, 1, len(clusters_loc)))

for c_id, col in zip(clusters_loc, colors_loc):
    mask = df["cluster_location"] == c_id
    plt.scatter(
        df.loc[mask, "lon_2022"],
        df.loc[mask, "lat_2022"],
        c=[col],
        s=10,
        alpha=0.9,
        label=f"Cluster {c_id}",
    )

plt.title("Clusters by Location (Spatial Zones)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(markerscale=2, fontsize=9, title="Location cluster")
plt.tight_layout()
plt.show()

# =====================================================
# 10. TABLES
# =====================================================

risk_summary = (
    df.groupby("risk_bucket")
      .agg(
          total_points=("predicted_clearance_2025", "size"),
          mean_clearance_ft=("predicted_clearance_2025", "mean"),
          min_clearance_ft=("predicted_clearance_2025", "min"),
          max_clearance_ft=("predicted_clearance_2025", "max"),
      )
      .reset_index()
)

risk_summary["percent_of_total"] = (
    100.0 * risk_summary["total_points"] / len(df)
)

print("\n=== TABLE A: Predicted clearance by risk bucket ===")
print(risk_summary.to_string(index=False))

risk_summary.to_csv("table_risk_by_clearance_bucket.csv", index=False)
print("Saved: table_risk_by_clearance_bucket.csv")

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
# CLUSTERING: Location + Predicted Growth (adjustable weights)
# =====================================================

n_clusters = 5
w_loc = 2.25
w_growth = 1.0

required_cols = ["lon_2022", "lat_2022", "predicted_growth_24_25"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns for clustering: {missing}")

X = df[required_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled[:, 0] *= w_loc
X_scaled[:, 1] *= w_loc
X_scaled[:, 2] *= w_growth

kmeans_loc_growth = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster_location_growth"] = kmeans_loc_growth.fit_predict(X_scaled)

print("Created df['cluster_location_growth'] using location + predicted growth clustering.")
print(f"w_loc={w_loc}, w_growth={w_growth}, n_clusters={n_clusters}")

# =====================================================
# MAP: One color per cluster (location + predicted growth)
# =====================================================

plt.figure(figsize=(10, 8))

cluster_ids = sorted(df["cluster_location_growth"].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))

for cid, col in zip(cluster_ids, colors):
    mask = df["cluster_location_growth"] == cid
    plt.scatter(
        df.loc[mask, "lon_2022"],
        df.loc[mask, "lat_2022"],
        c=[col],
        s=10,
        alpha=0.9,
        label=f"Cluster {cid}"
    )

plt.title("Clusters by Location + Predicted Growth (Adjustable Weights)")
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

cols_2025 = ["Latitude", "Longitude", "Clearance"]
df_2025 = pd.read_csv("capstone2025.csv", usecols=cols_2025)

df_2025 = df_2025.rename(columns={
    "Latitude": "lat_2025",
    "Longitude": "lon_2025",
    "Clearance": "clearance_2025_actual"
})
df_2025 = df_2025.dropna(subset=["lat_2025", "lon_2025", "clearance_2025_actual"])

df_2025["clearance_2025_actual"] = pd.to_numeric(df_2025["clearance_2025_actual"], errors="coerce")
df_2025 = df_2025.dropna(subset=["clearance_2025_actual"])

df_model = df.copy()

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

gdf_model_m = gdf_model.to_crs("EPSG:3395")
gdf_2025_m  = gdf_2025.to_crs("EPSG:3395")

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

if "index_y2025" in validation_join.columns:
    validation_join = validation_join.drop(columns=["index_y2025"])
elif "index_right" in validation_join.columns:
    validation_join = validation_join.drop(columns=["index_right"])

val = pd.DataFrame(validation_join.drop(columns="geometry")).copy()

if "predicted_clearance_2025" not in val.columns:
    raise ValueError("Missing predicted_clearance_2025. Make sure prediction step ran before validation.")

val["error_clearance_ft"] = val["predicted_clearance_2025"] - val["clearance_2025_actual"]
val["abs_error_clearance_ft"] = val["error_clearance_ft"].abs()

if "clearance_2024" in val.columns:
    val["actual_growth_24_25"] = val["clearance_2024"] - val["clearance_2025_actual"]
    val["error_growth_ft"] = val["predicted_growth_24_25"] - val["actual_growth_24_25"]
    val["abs_error_growth_ft"] = val["error_growth_ft"].abs()

if "actual_growth_24_25" not in val.columns:
    val["actual_growth_24_25"] = val["clearance_2024"] - val["clearance_2025_actual"]

before = len(val)
val = val[val["actual_growth_24_25"] > -1.0].copy()
after = len(val)

print(f"\nFiltered out {before - after} points with actual_growth_24_25 <= -3.0 ft (likely trimming).")
print(f"Remaining validation points: {after}")

mae = val["abs_error_clearance_ft"].mean()
rmse = np.sqrt((val["error_clearance_ft"]**2).mean())
bias = val["error_clearance_ft"].mean()

y_true = val["clearance_2025_actual"].values
y_pred = val["predicted_clearance_2025"].values
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

std_err_clearance = val["error_clearance_ft"].std()
std_abs_err_clearance = val["abs_error_clearance_ft"].std()
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

val.to_csv("validation_predicted_vs_actual_2025.csv", index=False)
print("Saved: validation_predicted_vs_actual_2025.csv")

# ---- G) Plots ----

plt.figure(figsize=(7, 6))
plt.scatter(val["clearance_2025_actual"], val["predicted_clearance_2025"], s=10, alpha=0.6)
minv = min(val["clearance_2025_actual"].min(), val["predicted_clearance_2025"].min())
maxv = max(val["clearance_2025_actual"].max(), val["predicted_clearance_2025"].max())
plt.plot([minv, maxv], [minv, maxv], linestyle="--")
plt.xlabel("Actual Clearance 2025 (ft)")
plt.ylabel("Predicted Clearance 2025 (ft)")
plt.title("Predicted vs Actual Clearance (2025)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(val["error_clearance_ft"].dropna(), bins=40, alpha=0.8)
plt.xlabel("Prediction Error (ft)  [Pred - Actual]")
plt.ylabel("Count")
plt.title("Clearance Prediction Error Distribution (2025)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sc = plt.scatter(
    val["lon_2024"], val["lat_2024"],
    c=val["abs_error_clearance_ft"],
    s=10,
    alpha=0.85,
    cmap="magma"
)
cbar = plt.colorbar(sc)
cbar.set_label("Absolute Error in Clearance (ft)")
plt.title("Spatial Error Map: |Predicted - Actual| Clearance (2025)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

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

required = ["predicted_growth_24_25", "actual_growth_24_25"]
missing = [c for c in required if c not in val.columns]
if missing:
    raise ValueError(f"Missing required columns for growth accuracy: {missing}")

val["abs_growth_error_ft"] = (
    val["predicted_growth_24_25"] - val["actual_growth_24_25"]
).abs()

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

threshold = 5.0

if "predicted_clearance_2025" not in val.columns or "clearance_2025_actual" not in val.columns:
    raise ValueError("Missing predicted_clearance_2025 or clearance_2025_actual in val.")

pred_danger = val["predicted_clearance_2025"] < threshold
actual_danger = val["clearance_2025_actual"] < threshold

TP = int((pred_danger & actual_danger).sum())
FP = int((pred_danger & ~actual_danger).sum())
FN = int((~pred_danger & actual_danger).sum())
TN = int((~pred_danger & ~actual_danger).sum())

total = len(val)

recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
miss_rate = FN / (TP + FN) if (TP + FN) > 0 else np.nan
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

missed = val[(val["clearance_2025_actual"] < threshold) & (val["predicted_clearance_2025"] >= threshold)].copy()
missed.to_csv("missed_danger_points_actual_under4_predicted_safe.csv", index=False)
print("Saved: missed_danger_points_actual_under4_predicted_safe.csv")

plt.figure(figsize=(10, 8))
plt.scatter(val["lon_2024"], val["lat_2024"], s=6, alpha=0.25, label="All validation points")
plt.scatter(missed["lon_2024"], missed["lat_2024"], s=18, alpha=0.9, label="MISSED danger (<4ft actual)")
plt.title("Missed Danger Points: Actual <4ft but Predicted >=4ft")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

