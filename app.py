import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import matplotlib.patches as mpatches

# Import your heavy-lifting math functions from your engine file
from model_engine import load_and_prepare_data, predict_and_cluster

st.set_page_config(page_title="FPL Vegetation Predictor", layout="wide")
st.title("🌴 VegeMan X Alpha")

# --- UI CONTROLS (SIDEBAR) ---
st.sidebar.header("📁 Data Input")
file_22 = st.sidebar.file_uploader("Upload 2022 Data", type=["csv"])
file_23 = st.sidebar.file_uploader("Upload 2023 Data", type=["csv"])
file_24 = st.sidebar.file_uploader("Upload 2024 Data", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Scenario Parameters")

# 1. Rain input (Text box / Number Input)
user_rain = st.sidebar.number_input("Predicted Rainfall (inches)", min_value=0.0, max_value=200.0, value=48.7, step=0.1)

# 2. w_loc slider
user_w_loc = st.sidebar.slider("Routing Weight (w_loc)", min_value=0.5, max_value=5.0, value=2.25, step=0.25)

# 3. Clearance Danger Threshold (0 to 20)
user_danger = st.sidebar.slider("Clearance Threshold (ft)", min_value=0.0, max_value=20.0, value=4.0, step=0.5)

# 4. Region Dropdown Menu
user_region = st.sidebar.selectbox("Select Region to Display", ["All Regions", "north", "southwest", "southeast"])

st.sidebar.markdown("---")

# --- EXECUTION BUTTON ---
if st.sidebar.button("🚀 Run Prediction Engine"):
    if file_22 and file_23 and file_24:
        with st.spinner("Processing LiDAR zones and running K-Means..."):
            
            # Run the heavy math (consolidated into one clean step)
            df_base = load_and_prepare_data(file_22, file_23, file_24)
            df_final = predict_and_cluster(df_base, user_rain, user_w_loc, 1.0, user_danger)
            
            # Save the final data into Streamlit's memory (Session State)
            st.session_state['df_final'] = df_final
            
    else:
        st.sidebar.error("⚠️ Please upload all three CSV files.")

# --- MAIN DISPLAY (Only renders AFTER the model is run) ---
if 'df_final' in st.session_state:
    
    # Grab the data from memory
    df_display = st.session_state['df_final'].copy()
    
    # Filter by the Dropdown selection
    if user_region != "All Regions":
        df_display = df_display[df_display['zone'] == user_region]
        st.success(f"Model Complete! Displaying data for the **{user_region.upper()}** zone.")
    else:
        st.success("Model Complete! Displaying data for **ALL REGIONS**.")

    # ==========================================
    # 🚨 DIAGNOSTIC TOOL 🚨
    # ==========================================
    st.write("📊 **Diagnostic Tool - Total points per zone in the raw dataset:**", st.session_state['df_final']['zone'].value_counts().to_dict())
    st.write(f"🔍 **Points currently being displayed ({user_region}):** {len(df_display)}")
    st.markdown("---")
    # ==========================================

    tab1, tab2 = st.tabs(["🗺️ Geospatial Risk Map", "📊 Operations Dashboard"])

    with tab1:
        st.header("2025 Predicted Clearance")
        
        # --- NATIVE UI LEGEND ---
        st.markdown("**Risk Level Legend (Predicted Distance to Wire):**")
        leg1, leg2, leg3, leg4, leg5 = st.columns(5)
        with leg1: st.markdown(f"🟥 **< {user_danger} ft** (Critical)")
        with leg2: st.markdown("🟧 **4–6 ft** (High Priority)")
        with leg3: st.markdown("🟨 **6–8 ft** (Watch List)")
        with leg4: st.markdown("🟩 **8–10 ft** (Healthy)")
        with leg5: st.markdown("⬜ **≥ 10 ft** (Safe)")
        st.markdown("---")

        # --- MAP GENERATION ---
        # If the region has no data, warn the user instead of crashing
        if df_display.empty:
            st.warning("No data available for the selected region.")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Reproject to Web Mercator for the satellite background
            gdf_plot = gpd.GeoDataFrame(
                df_display, 
                geometry=gpd.points_from_xy(df_display["lon_2022"], df_display["lat_2022"]), 
                crs="EPSG:4326"
            ).to_crs(epsg=3857)
            
            # Plot the points (s=25 makes them visible on high-res maps)
            ax.scatter(gdf_plot.geometry.x, gdf_plot.geometry.y, c=gdf_plot["risk_color"], s=25, alpha=0.9)
            
            # Lock the aspect ratio to prevent stretching
            ax.set_aspect('equal')
            
            # Add Satellite Background (Forced High-Definition)
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=14)
                
            st.pyplot(fig)


    with tab2:
        st.header("Risk Distribution Summary")
        if not df_display.empty:
            # Create a clean summary table based on the filtered data
            risk_summary = df_display.groupby("risk_bucket").size().reset_index(name="Point Count")
            st.dataframe(risk_summary, use_container_width=True)
        else:
            st.write("No data to summarize.")