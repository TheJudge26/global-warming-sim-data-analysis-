import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pycountry_convert as pc

from src.config import CONTINENTS
from src.api_service import get_live_country_data


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Supreme Climate AI",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Live Climate AI Predictor")
st.write("Select a country to simulate a world living exactly like that country.")

st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 26px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# CACHED MODEL LOADING
# =========================================================

@st.cache_resource
def load_models():
    """
    Loads only the Neural Network and Scaler. 
    (Prophet is bypassed because we read the CSV directly).
    """
    try:
        nn_model = joblib.load("models/supreme_nn_model.pkl")
        scaler = joblib.load("models/supreme_scaler.pkl")
        return nn_model, scaler

    except FileNotFoundError as e:
        st.error(f"❌ Missing model file: {e}")
        st.stop()

# Load them properly into the app (No 'm' variable anymore!)
nn_model, scaler = load_models()

@st.cache_data
def load_dataset():
    """
    Loads processed climate dataset.
    """
    try:
        return pd.read_csv("data/processed/supreme_dataset.csv")
    except FileNotFoundError:
        st.error("❌ supreme_dataset.csv not found.")
        st.stop()


@st.cache_data
def load_feature_importance():
    """
    Loads explainability feature importance.
    """
    try:
        return pd.read_csv("data/processed/feature_importance.csv")
    except FileNotFoundError:
        return pd.DataFrame()


# =========================================================
# LOAD RESOURCES
# =========================================================

nn_model, scaler = load_models()

supreme_df = load_dataset()

importance_df = load_feature_importance()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_continent(country_name):

    try:
        country_code = pc.country_name_to_country_alpha2(country_name)

        continent_code = pc.country_alpha2_to_continent_code(country_code)

        mapping = {
            "AF": "Africa",
            "AS": "Asia",
            "EU": "Europe",
            "NA": "North America",
            "SA": "South America",
            "OC": "Oceania"
        }

        return mapping.get(continent_code, "Other")

    except:
        return "Other"


def get_risk_level(value):

    if value < 1.0:
        return "🟢 Safe"

    elif value < 1.5:
        return "🟡 Moderate"

    else:
        return "🔴 Dangerous"


# =========================================================
# FEATURE COLUMNS
# =========================================================

FEATURE_COLUMNS = [
    "Year",
    "CO2_Emissions",
    "Population",
    "Total_GHG",
    "Temp_Moving_Avg",
    "temperature_anomaly_lag_1",
    "temperature_anomaly_lag_3",
    "temperature_anomaly_lag_5",
    "co2_lag_1",
    "rolling_temp_3",
    "rolling_temp_5",
    "rolling_co2_5",
    "temp_trend_5"
]


# =========================================================
# COUNTRY CONFIG (Dynamically Unlocked)
# =========================================================

# Extract all valid countries directly from the true science dataset!
valid_countries = supreme_df.dropna(subset=['iso_code', 'Real_Country_Name']).drop_duplicates(subset=['Real_Country_Name'])
COUNTRY_TO_ISO = dict(zip(valid_countries['Real_Country_Name'], valid_countries['iso_code']))
SUPPORTED_COUNTRIES = sorted(list(COUNTRY_TO_ISO.keys()))


# =========================================================
# DATA PREPARATION
# =========================================================

supreme_df["continent"] = supreme_df["Real_Country_Name"].apply(get_continent)

st.markdown("---")


# =========================================================
# USER INPUTS
# =========================================================

country_name_input = st.selectbox(
    "Select Country",
    SUPPORTED_COUNTRIES
)

region = st.selectbox(
    "Select Region (Optional)",
    ["Global"] + CONTINENTS
)


# =========================================================
# MAIN SIMULATION
# =========================================================

if st.button("Run Simulation"):

    with st.spinner(f"📡 Accessing satellite data for {country_name_input}..."):

        # =================================================
        # COUNTRY MAPPING
        # =================================================

        country_name = country_name_input
        country_code = COUNTRY_TO_ISO[country_name]
        country_continent = get_continent(country_name)


        # =================================================
        # FETCH LIVE DATA
        # =================================================

        live_df = get_live_country_data(country_code)

        if live_df.empty:
            st.error("❌ Failed to retrieve live data.")
            st.stop()


        # =================================================
        # REGION VALIDATION
        # =================================================

        if region != "Global" and country_continent != region:

            st.error(
                f"❌ Invalid selection: {country_name_input} belongs to "
                f"{country_continent}, not {region}"
            )

            st.stop()

        if region == "Global":
            region_df = supreme_df

        else:
            region_df = supreme_df[
                supreme_df["continent"] == region
            ]


        # =================================================
        # POPULATION SAFETY VALVE
        # =================================================

        country_pop = live_df['Population'].iloc[0]

        if country_pop <= 0:

            st.error(
                f"❌ API Error: Live population data missing for "
                f"{country_name_input}"
            )

            st.stop()


        # =================================================
        # HANDLE CO2 FALLBACK (Smart Historical Backup)
        # =================================================

        if (
            'CO2_Emissions' not in live_df.columns
            or live_df['CO2_Emissions'].iloc[0] == 0
        ):
            st.warning("⚠️ Live CO2 API lagging. Pulling the most recent profile from historical records...")
            
            # Look up the country in our supreme dataset
            country_history = supreme_df[supreme_df['iso_code'] == country_code]
            
            if not country_history.empty:
                # Get their most recent recorded year
                latest_row = country_history.sort_values('Year').iloc[-1]
                hist_co2_mt = latest_row['CO2_Emissions']
                hist_pop = latest_row['Population']
                
                # Calculate per capita in Million Tonnes (MT)
                per_capita_mt = hist_co2_mt / hist_pop if hist_pop > 0 else (4.5 / 1_000_000)
            else:
                # Global average fallback (4.5 tonnes per person = 0.0000045 MT)
                per_capita_mt = 4.5 / 1_000_000

            # Inject the calculated historical emissions into the live dataframe
            live_df['CO2_Emissions'] = live_df['Population'].iloc[0] * per_capita_mt

        # =================================================
        # DISPLAY LIVE DATA
        # =================================================

        st.success("✅ Data Fetched Successfully!")
        st.dataframe(live_df)

# 4. GLOBAL "WHAT IF" SIMULATION
        raw_api_co2 = live_df['CO2_Emissions'].iloc[0]
        
        # 🛡️ BULLETPROOF UNIT CHECK: 
        # If the API gave us raw tonnes (e.g. 5,000,000,000), convert to MT. 
        # If it's already MT (e.g. 5000), leave it alone.
        if raw_api_co2 > 1_000_000:
            country_co2_mt = raw_api_co2 / 1_000_000
        else:
            country_co2_mt = raw_api_co2

        per_capita_mt = country_co2_mt / country_pop
        global_population_2026 = 8000000000
        
        # Emissions for ONE year (in Million Tonnes)
        ai_co2_emissions = per_capita_mt * global_population_2026
        
        # Text display requires Raw Tonnes
        display_tonnes = ai_co2_emissions * 1_000_000

        st.info(f"🔍 **Simulation:** If the entire planet emitted carbon like **{country_code}**, global CO2 would be **{display_tonnes:,.0f} tonnes per year**.")

        # 5. FETCH HISTORICAL CUMULATIVE CO2
        last_year = supreme_df['Year'].max()
        if 'Cumulative_CO2' in supreme_df.columns:
            current_global_cumulative = supreme_df[supreme_df['Year'] == last_year]['Cumulative_CO2'].sum()
        else:
            current_global_cumulative = supreme_df['CO2_Emissions'].sum()

        # 🚀 THE 10-YEAR PHYSICS FIX
        simulated_global_cumulative = current_global_cumulative + (ai_co2_emissions * 10)

        # 6. SUPREME AI PREDICTION
        ai_input = pd.DataFrame([[
            2036, 
            ai_co2_emissions, 
            simulated_global_cumulative, 
            global_population_2026
        ]], columns=['Year', 'CO2_Emissions', 'Cumulative_CO2', 'Population'])
            
        X_scaled = scaler.transform(ai_input)
        prediction = nn_model.predict(X_scaled)[0]
        risk = get_risk_level(prediction)



        # =================================================
        # RESULTS
        # =================================================

        st.markdown("---")

        st.subheader("🤖 AI Global Temperature Prediction")

        st.metric(
            "Climate Risk Level",
            risk
        )

        if prediction > 1.5:

            st.error(
                f"🌡️ Catastrophic Warning: "
                f"Projected Temperature = "
                f"+{prediction:.2f}°C"
            )

        else:

            st.success(
                f"🌤️ Safe Zone: "
                f"Projected Temperature = "
                f"+{prediction:.2f}°C"
            )


        # =================================================
        # GAUGE CHART (Zoomed in for scientific sensitivity)
        # =================================================

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,

            title={'text': "Global Temperature Rise (°C)"},

            gauge={
                'axis': {'range': [1.0, 2.5]}, # 🚀 ZOOMED IN!

                'steps': [
                    {'range': [1.0, 1.4], 'color': "lightgreen"},
                    {'range': [1.4, 1.6], 'color': "yellow"},
                    {'range': [1.6, 2.0], 'color': "orange"},
                    {'range': [2.0, 2.5], 'color': "red"}
                ],
                'bar': {'color': "black"} # Makes the needle easier to see
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # =================================================
        # EXPLAINABLE AI
        # =================================================

        st.markdown("---")

        st.subheader(
            "🔍 What is driving the prediction?"
        )

        if not importance_df.empty:

            fig_imp = px.bar(
                importance_df,
                x="Feature",
                y="Importance",
                title="Feature Impact on Climate Prediction"
            )

            st.plotly_chart(
                fig_imp,
                use_container_width=True
            )


        # =================================================
        # FORECASTING
        # =================================================


        st.subheader("📈 Future Climate Forecast (20 Years)")

        # 🚀 THE FIX: Read the pre-calculated forecast from your data pipeline!
        forecast = pd.read_csv("data/processed/future_forecast.csv")
        
        # Find the last known year for the boundary line
        last_year = supreme_df['Year'].max()

        fig_forecast = px.line(
            forecast,
            x='ds',
            y='yhat',
            title='20-Year Global Temperature Forecast',
            labels={
                "ds": "Year",
                "yhat": "Temperature Anomaly (°C)"
            }
        )

        fig_forecast.add_vline(
            x=f"{last_year}-01-01",
            line_width=2,
            line_dash="dash",
            line_color="red"
        )

        st.plotly_chart(
            fig_forecast,
            use_container_width=True
        )


        # =================================================
        # COUNTRY SAFETY RANKING
        # =================================================

        st.subheader("🌍 Global Climate Safety Ranking")

        ranking_df = supreme_df.groupby(
            "Real_Country_Name",
            as_index=False
        ).agg({
            "Average_Temperature": "mean",
            "CO2_Emissions": "mean",
            "Total_GHG": "mean"
        })

        ranking_df["risk_score"] = (
            ranking_df["CO2_Emissions"] * 0.4 +
            ranking_df["Total_GHG"] * 0.3 +
            ranking_df["Average_Temperature"] * 0.3
        )

        ranking_df["risk_score"] = (
            ranking_df["risk_score"].round(2)
        )

        ranking_df = ranking_df.sort_values(
            "risk_score"
        )

        st.dataframe(ranking_df.head(10))


        # =================================================
        # REGIONAL ANALYTICS
        # =================================================

        st.subheader(
            f"🌍 Regional Climate Overview: {region}"
        )

        region_summary = (
            region_df
            .groupby("continent")["CO2_Emissions"]
            .mean()
            .reset_index()
        )

        fig_region = px.bar(
            region_summary,
            x="continent",
            y="CO2_Emissions",
            title="Average CO2 Emissions by Region"
        )

        st.plotly_chart(
            fig_region,
            use_container_width=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Intelligence provided by OWID, "
    "World Bank, and Supreme Neural Networks"
)

st.markdown("---")