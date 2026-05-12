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
    Loads all AI models and scalers only once.
    """

    try:
        lgbm_model = joblib.load("models/supreme_lgbm_model.pkl")
        scaler = joblib.load("models/supreme_scaler.pkl")
        prophet_model = joblib.load("models/prophet_model.pkl")

        return lgbm_model, scaler, prophet_model

    except FileNotFoundError as e:
        st.error(f"❌ Missing model file: {e}")
        st.stop()


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

lgbm_model, scaler, m = load_models()

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
# COUNTRY CONFIG
# =========================================================

COUNTRY_TO_ISO = {
    "Egypt": "EGY",
    "United States": "USA",
    "Brazil": "BRA",
    "Canada": "CAN",
    "India": "IND",
    "China": "CHN",
    "Australia": "AUS",
    "Russia": "RUS"
}

SUPPORTED_COUNTRIES = [
    "Egypt",
    "United States",
    "Brazil",
    "Canada",
    "India",
    "China",
    "Australia",
    "Russia"
]

DISPLAY_TO_DATASET = {
    "Egypt": "Egypt",
    "United States": "USA",
    "Brazil": "Brazil",
    "Canada": "Canada",
    "India": "India",
    "China": "China",
    "Australia": "Australia",
    "Russia": "Russia"
}


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

        country_name = DISPLAY_TO_DATASET[country_name_input]

        country_code = COUNTRY_TO_ISO.get(country_name_input)

        country_continent = get_continent(country_name_input)


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
        # HANDLE CO2 FALLBACK
        # =================================================

        if (
            'CO2_Emissions' not in live_df.columns
            or live_df['CO2_Emissions'].iloc[0] == 0
        ):

            try:

                with open('models/carbon_profiles.json', 'r') as f:
                    profiles = json.load(f)

                multiplier = profiles.get(country_name, 4.5)

                st.warning(
                    f"⚠️ Live CO2 lagging. "
                    f"Using historical profile: "
                    f"{multiplier:.2f} tons/capita."
                )

                live_df['CO2_Emissions'] = (
                    live_df['Population'] * multiplier
                ) / 1000

            except FileNotFoundError:

                st.error("❌ carbon_profiles.json missing.")
                st.stop()


        # =================================================
        # DISPLAY LIVE DATA
        # =================================================

        st.success("✅ Data Fetched Successfully!")
        st.dataframe(live_df)


        # =================================================
        # GLOBAL WHAT-IF SIMULATION
        # =================================================

        country_co2_tonnes = (
            live_df['CO2_Emissions'].iloc[0] * 1000
        )

        per_capita_co2 = (
            country_co2_tonnes / country_pop
        )

        global_population_2026 = 8_000_000_000

        simulated_global_co2 = (
            per_capita_co2 * global_population_2026
        )

        st.info(
            f"🔍 Simulation: If the entire planet emitted "
            f"carbon like {country_name_input}, "
            f"global CO2 would become "
            f"{simulated_global_co2:,.0f} tonnes."
        )


        # =================================================
        # HISTORICAL GHG
        # =================================================

        country_history = supreme_df[
            supreme_df['Real_Country_Name'] == country_name
        ]

        if not country_history.empty:

            latest_row = (
                country_history
                .sort_values('Year')
                .iloc[-1]
            )

            hist_ghg = latest_row['Total_GHG']
            hist_pop = latest_row['Population']

            ghg_per_capita = hist_ghg / (
                hist_pop if hist_pop > 0 else 1
            )

            simulated_global_ghg = (
                ghg_per_capita * global_population_2026
            )

        else:
            simulated_global_ghg = 0


        # =================================================
        # FEATURE ENGINEERING
        # =================================================

        country_rows = supreme_df[
            supreme_df['Real_Country_Name'] == country_name
        ]

        if country_rows.empty:
            st.error(f"No historical data for {country_name}")
            st.stop()

        latest_features = (
            country_rows
            .sort_values("Year")
            .iloc[-1]
        )

        ai_input = pd.DataFrame([{
            "Year": 2026,
            "CO2_Emissions": simulated_global_co2,
            "Population": global_population_2026,
            "Total_GHG": simulated_global_ghg,

            "Temp_Moving_Avg":
                latest_features["Temp_Moving_Avg"],

            "temperature_anomaly_lag_1":
                latest_features["temperature_anomaly_lag_1"],

            "temperature_anomaly_lag_3":
                latest_features["temperature_anomaly_lag_3"],

            "temperature_anomaly_lag_5":
                latest_features["temperature_anomaly_lag_5"],

            "co2_lag_1":
                latest_features["co2_lag_1"],

            "rolling_temp_3":
                latest_features["rolling_temp_3"],

            "rolling_temp_5":
                latest_features["rolling_temp_5"],

            "rolling_co2_5":
                latest_features["rolling_co2_5"],

            "temp_trend_5":
                latest_features["temp_trend_5"]
        }])


        # =================================================
        # AI PREDICTION
        # =================================================

        X_scaled = scaler.transform(
            ai_input[FEATURE_COLUMNS]
        )

        prediction = lgbm_model.predict(X_scaled)[0]

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
        # GAUGE CHART
        # =================================================

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,

            title={'text': "Global Temperature Rise"},

            gauge={
                'axis': {'range': [0, 4]},

                'steps': [
                    {'range': [0, 1], 'color': "lightgreen"},
                    {'range': [1, 1.5], 'color': "yellow"},
                    {'range': [1.5, 2], 'color': "orange"},
                    {'range': [2, 4], 'color': "red"}
                ]
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

        st.subheader(
            "📈 Future Climate Forecast (20 Years)"
        )

        last_year = m.history['ds'].dt.year.max()

        future = m.make_future_dataframe(
            periods=20,
            freq='YE'
        )

        forecast = m.predict(future)

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
            x=pd.Timestamp(str(last_year)),
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