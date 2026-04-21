import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx

# Import the math engine
from model_engine import load_and_prepare_data, predict_and_cluster

st.set_page_config(page_title="FPL Vegetation Predictor", layout="wide")

# CSS hack to completely hide the +/- buttons on number inputs for a cleaner UI (this doesn't work :( )
st.markdown(
    """
    <style>
        button[title="Step down"] {display: none;}
        button[title="Step up"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌴 VegeMan X Beta")

# ==========================================
# UI CONTROLS (SIDEBAR)
# ==========================================
st.sidebar.header("📁 Dynamic Data Input")
num_years = st.sidebar.number_input("How many years of historical data?", min_value=2, max_value=10, value=3)

uploaded_files = []
for i in range(num_years):
    file = st.sidebar.file_uploader(f"Upload Year {i+1} Data (CSV)", type=["csv"], key=f"file_{i}")
    if file:
        uploaded_files.append(file)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Scenario Parameters")
st.sidebar.subheader("🌧️ Rainfall Parameters (inches)")

historical_rains = []
for i in range(num_years - 1):
    rain = st.sidebar.number_input(f"Historical Rain: Year {i+1} ➔ Year {i+2}", value=50.0, step=None, key=f"rain_{i}")
    historical_rains.append(rain)

rain_forecast = st.sidebar.number_input("Desired Year Forecast Rain", value=48.7, step=None)

st.sidebar.markdown("---")
user_danger = st.sidebar.slider("Danger Threshold (ft)", min_value=0.0, max_value=20.0, value=4.0, step=0.5)

# EXECUTION BUTTON
if st.sidebar.button("🚀 Run Prediction Engine"):
    if len(uploaded_files) == num_years:
        with st.spinner("Processing dynamic LiDAR data and running K-Nearest Neighbors..."):
            
            # OPEN THE FILES: Only load the essential columns into RAM to prevent OOM crashes.
            keep_cols = [
                'substation', 'line.type', 'latitude', 'longitude', 'clearance',
                'encroachment.length.u0', 'encroachment.length.u1', 
                'encroachment.length.u2', 'encroachment.length.u3', 
                'encroachment.length.u4'
            ]
            
            raw_dfs = [
                pd.read_csv(f, usecols=lambda x: x.strip().lower() in keep_cols) 
                for f in uploaded_files
            ]
            
            # RUN THE MATH: Hand the optimized DataFrames to the math engine
            df_base = load_and_prepare_data(raw_dfs)
            raw_output = predict_and_cluster(df_base, historical_rains, rain_forecast, user_w_loc, 1.0, user_danger)
            
            if isinstance(raw_output, (tuple, list)):
                df_final = raw_output[0]
            else:
                df_final = raw_output
            
            st.session_state['df_final'] = df_final
            st.rerun() # Refresh app to reveal filters
    else:
        st.sidebar.error(f"⚠️ Please upload all {num_years} CSV files before running.")


# MAIN DISPLAY & DYNAMIC FILTERS
if 'df_final' in st.session_state:
    
    df_display = st.session_state['df_final'].copy()

    # --- DYNAMIC FILTERS ---
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Display Filters")
    
    # Substation Filter (Checks if column exists first)
    if 'substation' in df_display.columns:
        available_substations = ["All Substations"] + sorted(df_display['substation'].dropna().unique().tolist())
        user_substation = st.sidebar.selectbox("Filter by Substation", available_substations)
        if user_substation != "All Substations":
            df_display = df_display[df_display['substation'] == user_substation]
    
    # Line Type Filter (Checks if column exists first)
    if 'line_type' in df_display.columns:
        available_lines = ["All Lines"] + sorted(df_display['line_type'].dropna().unique().tolist())
        user_line_type = st.sidebar.selectbox("Filter by Line Type", available_lines)
        if user_line_type != "All Lines":
            df_display = df_display[df_display['line_type'] == user_line_type]

    st.success("✅ Model Complete! Operations Dashboards are live.")

    # --- DIAGNOSTIC TOOL ---
    st.write(f"📊 **Total 60x30 Work Zones Tracked:** {len(st.session_state['df_final'])}")
    st.write(f"🔍 **Currently Displaying (Filtered):** {len(df_display)}")
    st.markdown("---")

    # --- DASHBOARD TABS ---
    tab1, tab2 = st.tabs(["🗺️ Geospatial Risk Map", "📊 Operations Dashboard"])

    with tab1:
        st.header("Predicted Clearance Risk Map")
        
        # Native UI Legend based on user's threshold
        st.markdown("**Risk Level Legend (Predicted Distance to Wire):**")
        leg1, leg2, leg3, leg4, leg5 = st.columns(5)
        with leg1: st.markdown(f"🟥 **< {user_danger} ft** (Critical)")
        with leg2: st.markdown(f"🟧 **{user_danger}–{user_danger+2} ft** (High Priority)")
        with leg3: st.markdown(f"🟨 **{user_danger+2}–{user_danger+4} ft** (Watch List)")
        with leg4: st.markdown("🟩 **Healthy**")
        with leg5: st.markdown("⬜ **Safe**")
        st.markdown("---")

        if df_display.empty:
            st.warning("No data available for the selected filters.")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            gdf_plot = gpd.GeoDataFrame(
                df_display, 
                geometry=gpd.points_from_xy(df_display["lon"], df_display["lat"]), 
                crs="EPSG:4326"
            ).to_crs(epsg=3857)
            
            ax.scatter(gdf_plot.geometry.x, gdf_plot.geometry.y, c=gdf_plot["risk_color"], s=25, alpha=0.9)
            ax.set_aspect('equal')
            
            # Forced High-Definition Satellite Zoom
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=14)
            ax.set_axis_off()
            st.pyplot(fig)

        # --- PREDICTED VEGETATION DOWNLOAD ---
        st.markdown("---")
        st.subheader("📋 Export Prediction Data")
        
        @st.cache_data
        def convert_df(df):
            # We drop 'geometry' and 'risk_color' for the CSV to keep it clean for Excel/GIS
            cols_to_drop = [c for c in ['geometry', 'risk_color'] if c in df.columns]
            return df.drop(columns=cols_to_drop).to_csv(index=False).encode('utf-8')

        # This captures the current filtered state of the model (Substation, Line Type, etc.)
        prediction_csv = convert_df(df_display)

        st.download_button(
            label="📥 Download Predicted Vegetation (CSV)",
            data=prediction_csv,
            file_name='predicted_vegetation_export.csv',
            mime='text/csv',
        )

    with tab2:
        st.header("Risk Distribution Summary")
        if not df_display.empty:
            risk_summary = df_display.groupby("risk_bucket").size().reset_index(name="Work Zones")
            st.dataframe(risk_summary, use_container_width=True)
            
            if 'dispatch_cluster' in df_display.columns:
                st.subheader("Dispatch Clusters (K-Means Routing)")
                cluster_counts = df_display.groupby("dispatch_cluster").size().reset_index(name="Zones to Clear")
                st.dataframe(cluster_counts, use_container_width=True)
        else:
            st.write("No data to summarize.")
