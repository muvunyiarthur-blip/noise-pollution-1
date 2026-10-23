from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Noise Monitoring Dashboard", page_icon="🔊", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [ROOT / "data" / "noise_data.csv", ROOT / "noise_data.csv"]
MODEL_CANDIDATES = [ROOT / "models" / "kmeans_noise_model.joblib", ROOT / "kmeans_noise_model.joblib"]
FEATURE_COLUMNS = ["estimated_level_dbfs", "peak_amplitude", "zero_crossing_rate", "spectral_centroid_hz"]
CATEGORY_ORDER = ["Quiet", "Moderate", "Noisy"]

@st.cache_data
def load_data(path_string):
    df = pd.read_csv(path_string)
    for col in ["timestamp", "location", "hour", "rms", "peak_amplitude", "zero_crossing_rate",
                "spectral_centroid_hz", "estimated_level_dbfs", "cluster_id", "noise_category"]:
        if col not in df.columns:
            df[col] = np.nan
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["location"] = df["location"].fillna("Unspecified").astype(str)
    for col in ["hour", "rms", "peak_amplitude", "zero_crossing_rate", "spectral_centroid_hz",
                "estimated_level_dbfs", "cluster_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)

data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)

st.title("🔊 Noise Monitoring Dashboard")
st.caption("K-Means clustering • Quiet / Moderate / Noisy • Acoustic feature analysis")

if data_path is None:
    st.error("Dataset not found. Add noise_data.csv to the data/ folder or project root.")
    st.stop()

df = load_data(str(data_path))
if df.empty:
    st.warning("The dataset is empty. Add measurements to continue.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    locations = sorted(df["location"].dropna().unique().tolist())
    chosen_locations = st.multiselect("Location", locations, default=locations)
    categories = [c for c in CATEGORY_ORDER if c in df["noise_category"].dropna().unique()]
    chosen_categories = st.multiselect("Noise category", categories, default=categories)
    if df["timestamp"].notna().any():
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        date_range = None

filtered = df[df["location"].isin(chosen_locations)]
if chosen_categories:
    filtered = filtered[filtered["noise_category"].isin(chosen_categories)]
else:
    filtered = filtered.iloc[0:0]
if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = filtered[
        filtered["timestamp"].isna() |
        ((filtered["timestamp"].dt.date >= date_range[0]) & (filtered["timestamp"].dt.date <= date_range[1]))
    ]

total = len(filtered)
avg_db = filtered["estimated_level_dbfs"].mean()
noisy_count = int((filtered["noise_category"] == "Noisy").sum())
location_count = filtered["location"].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Measurements", f"{total:,}")
c2.metric("Average estimated level", "—" if pd.isna(avg_db) else f"{avg_db:.2f} dBFS")
c3.metric("Noisy readings", f"{noisy_count:,}")
c4.metric("Locations", f"{location_count:,}")

st.info("dBFS is a digital signal level, not calibrated acoustic dB SPL. Categories are unsupervised clusters ordered by their average estimated level.")

tab_overview, tab_locations, tab_data, tab_predict, tab_about = st.tabs(
    ["Overview", "Locations", "Measurement data", "Try a prediction", "Model & notes"]
)

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Measurements by noise category")
        counts = filtered["noise_category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
        st.bar_chart(counts)
    with right:
        st.subheader("Estimated level over time")
        timeline = filtered.dropna(subset=["timestamp", "estimated_level_dbfs"]).sort_values("timestamp")
        if not timeline.empty:
            st.line_chart(timeline.set_index("timestamp")["estimated_level_dbfs"])
        else:
            st.write("No timestamped measurements available for this selection.")
    st.subheader("Feature distributions")
    feature = st.selectbox("Choose feature", ["estimated_level_dbfs", "peak_amplitude", "zero_crossing_rate", "spectral_centroid_hz", "rms"])
    feature_data = filtered[feature].dropna()
    if not feature_data.empty:
        st.bar_chart(feature_data.value_counts(bins=20, sort=False).sort_index())
    else:
        st.write("No values available for this feature.")

with tab_locations:
    st.subheader("Average estimated level by location")
    location_summary = (filtered.dropna(subset=["estimated_level_dbfs"])
                        .groupby("location")["estimated_level_dbfs"].mean().sort_values())
    if not location_summary.empty:
        st.bar_chart(location_summary)
        st.dataframe(location_summary.rename("Average dBFS").reset_index(), use_container_width=True)
    else:
        st.write("No location-level measurements available.")
    st.subheader("Category mix by location")
    if not filtered.empty:
        mix = pd.crosstab(filtered["location"], filtered["noise_category"])
        st.dataframe(mix, use_container_width=True)

with tab_data:
    st.subheader("Filtered measurement records")
    st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_noise_data.csv", mime="text/csv")

with tab_predict:
    st.subheader("Classify acoustic features with the saved model")
    st.write("Enter feature values from a measurement. This predicts a cluster category; it does not record audio.")
    if model_path is None:
        st.warning("Saved model not found. Run the training notebook first and place the model in models/.")
    else:
        bundle = load_model(str(model_path))
        default_row = filtered.dropna(subset=FEATURE_COLUMNS).tail(1)
        if default_row.empty:
            default_row = df.dropna(subset=FEATURE_COLUMNS).tail(1)
        row = default_row.iloc[0] if not default_row.empty else None
        defaults = {
            "estimated_level_dbfs": float(row["estimated_level_dbfs"]) if row is not None else -45.0,
            "peak_amplitude": float(row["peak_amplitude"]) if row is not None else 0.05,
            "zero_crossing_rate": float(row["zero_crossing_rate"]) if row is not None else 0.10,
            "spectral_centroid_hz": float(row["spectral_centroid_hz"]) if row is not None else 1500.0,
        }
        with st.form("predict_form"):
            p1, p2 = st.columns(2)
            level = p1.number_input("Estimated level (dBFS)", min_value=-160.0, max_value=0.0, value=defaults["estimated_level_dbfs"], step=1.0)
            peak = p2.number_input("Peak amplitude", min_value=0.0, max_value=1.0, value=max(0.0, min(1.0, defaults["peak_amplitude"])), format="%.6f")
            zcr = p1.number_input("Zero-crossing rate", min_value=0.0, max_value=1.0, value=max(0.0, min(1.0, defaults["zero_crossing_rate"])), format="%.6f")
            centroid = p2.number_input("Spectral centroid (Hz)", min_value=0.0, max_value=8000.0, value=max(0.0, min(8000.0, defaults["spectral_centroid_hz"])), step=10.0)
            submitted = st.form_submit_button("Predict category")
        if submitted:
            X = pd.DataFrame([{
                "estimated_level_dbfs": level,
                "peak_amplitude": peak,
                "zero_crossing_rate": zcr,
                "spectral_centroid_hz": centroid
            }], columns=bundle.get("feature_columns", FEATURE_COLUMNS))
            cluster_id = int(bundle["pipeline"].predict(X)[0])
            category = bundle["cluster_to_category"][cluster_id]
            st.success(f"Predicted category: {category} (cluster {cluster_id})")
            st.caption("K-Means is unsupervised. This is a cluster assignment, not a probability or a supervised accuracy score.")

with tab_about:
    st.subheader("How the model works")
    st.markdown("""
    1. Each audio window is represented by four numeric features: estimated level (dBFS), peak amplitude, zero-crossing rate, and spectral centroid.
    2. `StandardScaler` standardizes the features so their different units do not dominate distance calculations.
    3. K-Means groups measurements into **three clusters**.
    4. The clusters are ordered by their mean estimated level and named **Quiet**, **Moderate**, and **Noisy**.
    5. The trained pipeline and cluster-to-category mapping are stored in a Joblib file.
    """)
    if model_path is not None:
        bundle = load_model(str(model_path))
        st.write("**Model trained at:**", bundle.get("trained_at", "Not recorded"))
        st.write("**Features:**", ", ".join(bundle.get("feature_columns", FEATURE_COLUMNS)))
        st.write("**Saved model:**", str(model_path.relative_to(ROOT)))
    st.warning("Cluster names describe relative groups in this dataset. They are not calibrated safety thresholds. The model's category labels can change when retrained on new data.")
    st.markdown("**Public deployment privacy:** this dashboard reads the repository's committed CSV and model. Do not commit private recordings, identifying location data, or confidential measurements.")
