import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import pycountry_convert as pc
import numpy as np
from src.config import CONTINENTS
from sklearn.preprocessing import MinMaxScaler
from src.api_service import get_live_country_data

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Supreme Climate ",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Live Anomaly Predictor")
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
# HELPER FUNCTIONS
# =========================================================

def get_continent(country_name):
    try:
        country_code = pc.country_name_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        mapping = {
            "AF": "Africa", "AS": "Asia", "EU": "Europe",
            "NA": "North America", "SA": "South America", "OC": "Oceania"
        }
        return mapping.get(continent_code, "Other")
    except:
        return "Other"

def get_risk_level(value):
    if value < 1.0: return "🟢 Safe"
    elif value < 1.5: return "🟡 Moderate"
    else: return "🔴 Dangerous"

# =========================================================
# CACHED MODEL & DATA LOADING
# =========================================================

@st.cache_resource
def load_models():
    try:
        nn_model = joblib.load("models/supreme_nn_model.pkl")
        scaler = joblib.load("models/supreme_scaler.pkl")
        return nn_model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Missing model file: {e}")
        st.stop()

nn_model, scaler = load_models()

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("data/processed/supreme_dataset.csv")
        
        df["continent"] = df["Real_Country_Name"].apply(get_continent)
        return df
    except FileNotFoundError:
        st.error("❌ supreme_dataset.csv not found.")
        st.stop()

@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv("data/processed/feature_importance.csv")
    except FileNotFoundError:
        return pd.DataFrame()

supreme_df = load_dataset()
importance_df = load_feature_importance()

@st.cache_data(ttl=3600)
def fetch_live_data_cached(country_code):
    return get_live_country_data(country_code)

# =========================================================
# COUNTRY CONFIG
# =========================================================

valid_countries = supreme_df.dropna(subset=['iso_code', 'Real_Country_Name']).drop_duplicates(subset=['Real_Country_Name'])
COUNTRY_TO_ISO = dict(zip(valid_countries['Real_Country_Name'], valid_countries['iso_code']))
SUPPORTED_COUNTRIES = sorted(list(COUNTRY_TO_ISO.keys()))

st.markdown("---")


# =================================================
# USER INPUTS & TIME MACHINE
# =================================================

st.markdown("### 🌍 Select Simulation Target")
col1, col2 = st.columns(2)
        
with col1:
    
    region = st.selectbox("Select Region Filter", ["Global"] + CONTINENTS)
    
    
    if region == "Global":
        filtered_countries = SUPPORTED_COUNTRIES
    else:
        region_df = valid_countries[valid_countries['continent'] == region]
        filtered_countries = sorted(region_df['Real_Country_Name'].tolist())
        
    country_name_input = st.selectbox("Select Country to Mimic", filtered_countries)
    
    st.markdown("###  The Time Machine")
    target_year = st.slider(
        "Simulate forward to year:", 
        min_value=2026, max_value=2100, value=2036, step=1
    )
    
with col2:
    st.markdown("###  Climate Policy Toggles")
    st.write("Apply global policies to see if we can beat the AI prediction:")
    ev_policy = st.checkbox("🔋 Global EV Revolution (-20% Emissions)")
    tax_policy = st.checkbox("🏭 Aggressive Carbon Tax (-10% Emissions)")


# =================================================
# MAIN SIMULATION
# =================================================

if st.button(" Run Simulation"):

    with st.spinner(f"📡 Accessing satellite data for {country_name_input}..."):

        country_name = country_name_input
        country_code = COUNTRY_TO_ISO[country_name]

        
        country_history = supreme_df[supreme_df['iso_code'] == country_code]
        
        
        if not country_history.empty:
            latest_row = country_history.sort_values('Year').iloc[-1]
            hist_pop = latest_row['Population']
            per_capita_co2 = latest_row['CO2_Emissions'] / hist_pop if hist_pop > 0 else (4.5 / 1_000_000)
            per_capita_ch4 = latest_row['Methane_Emissions'] / hist_pop if hist_pop > 0 else 0
            per_capita_n2o = latest_row['Nitrous_Oxide_Emissions'] / hist_pop if hist_pop > 0 else 0
        else:
            per_capita_co2 = 4.5 / 1_000_000
            per_capita_ch4 = 0
            per_capita_n2o = 0

       
        if ev_policy: 
            per_capita_co2 *= 0.80 
            per_capita_ch4 *= 0.95 
        if tax_policy: 
            per_capita_co2 *= 0.90
            per_capita_ch4 *= 0.90 
            per_capita_n2o *= 0.90 

        global_population_2026 = 8000000000
        
        ai_co2_emissions = per_capita_co2 * global_population_2026
        ai_ch4_emissions = per_capita_ch4 * global_population_2026
        ai_n2o_emissions = per_capita_n2o * global_population_2026
        
        
        display_gt = ai_co2_emissions / 1000 

        years_to_simulate = target_year - 2026
        if years_to_simulate < 1: years_to_simulate = 1

        st.info(f" **Simulation:** If the world acts like **{country_code}** until {target_year}, emitting **{display_gt:.1f} Gt of CO₂ per year**...")
        
        last_year = int(supreme_df['Year'].max())
        if 'Cumulative_CO2' in supreme_df.columns:
            current_global_cumulative = supreme_df[supreme_df['Year'] == last_year]['Cumulative_CO2'].sum()
        else:
            current_global_cumulative = supreme_df['CO2_Emissions'].sum()

        simulated_global_cumulative = current_global_cumulative + (ai_co2_emissions * years_to_simulate)
        log_simulated_cumulative = np.log1p(simulated_global_cumulative)

        ai_input = pd.DataFrame([[
            target_year, 
            ai_co2_emissions, 
            log_simulated_cumulative, 
            global_population_2026,
            ai_ch4_emissions, 
            ai_n2o_emissions  
        ]], columns=['Year', 'CO2_Emissions', 'Log_Cumulative_CO2', 'Population', 'Methane_Emissions', 'Nitrous_Oxide_Emissions'])
            
        X_scaled = scaler.transform(ai_input)
        future_predictions, future_std = nn_model.predict(X_scaled, return_std=True)
        prediction = future_predictions[0]
        risk = get_risk_level(prediction)

        years_beyond_target = max(0, target_year - last_year)
        target_penalty = 0.02 * (years_beyond_target / 10)
        honest_target_std = future_std[0] + target_penalty

        st.markdown("---")
        st.subheader(" Global Temperature Prediction")
        st.caption(" **Model Confidence (Historical Testing):** Training MAE (in-sample) of **±0.12°C**")
        
        col1, col2 = st.columns(2)
        with col1:
            std_95 = 1.96 * honest_target_std
            lower_val = prediction - std_95
            upper_val = prediction + std_95

            
            
            st.metric("Climate Risk Level", risk, delta=f"± {std_95:.2f}°C Margin of Error", delta_color="off")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': f"Simulated Temp Anomaly (°C)<br><span style='font-size:0.8em;color:gray'>95% CI: [{lower_val:.2f}°C to {upper_val:.2f}°C]</span>"},
                gauge={
                    'axis': {'range': [0.5, 3.0]}, 
                    'steps': [
                        {'range': [0.5, 1.5], 'color': "lightgreen"},
                        {'range': [1.5, 2.0], 'color': "orange"},
                        {'range': [2.0, 3.0], 'color': "red"}
                    ],
                    'bar': {'color': "black"}
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if prediction > 1.5:
                st.error(f"🌡️ Catastrophic Warning: +{prediction:.2f}°C")
            else:
                st.success(f"🌤️ Safe Zone: +{prediction:.2f}°C")
        
        with col2:
            ipcc_temps = [1.0, 1.5, 2.0, 4.0]
            ipcc_sea_levels = [0.1, 0.3, 0.5, 1.0]
            sea_level_rise = np.interp(prediction, ipcc_temps, ipcc_sea_levels)
            
            if sea_level_rise > 0.5:
                st.warning(f"🌊 **Coastal Impact Projection:** Expect roughly **{sea_level_rise:.2f} meters** of sea-level rise...")
            elif sea_level_rise >= 0.3:
                st.info(f"🌊 **Coastal Impact Projection:** Expect **{sea_level_rise:.2f} meters** of sea-level rise...")
            else:
                st.success(f"🌊 **Coastal Impact Projection:** Limited to **{sea_level_rise:.2f} meters**...")

        st.markdown("---")
        st.subheader(f"📈 Simulated Trajectory: {country_name} Path")
        
        future_years = list(range(last_year + 1, target_year + 1))
        
        if future_years:
            future_df = pd.DataFrame({'Year': future_years})
            future_df['CO2_Emissions'] = ai_co2_emissions
            future_df['Population'] = global_population_2026 
            future_df['Methane_Emissions'] = ai_ch4_emissions
            future_df['Nitrous_Oxide_Emissions'] = ai_n2o_emissions
            
            cumulative_list = []
            current_cum = current_global_cumulative
            for _ in future_years:
                current_cum += ai_co2_emissions
                cumulative_list.append(np.log1p(current_cum))
            future_df['Log_Cumulative_CO2'] = cumulative_list
            
            X_future = future_df[['Year', 'CO2_Emissions', 'Log_Cumulative_CO2', 'Population', 'Methane_Emissions', 'Nitrous_Oxide_Emissions']]
            X_future_scaled = scaler.transform(X_future)
            
            future_preds, base_std = nn_model.predict(X_future_scaled, return_std=True)
            
          
            years_beyond = np.array(future_years) - last_year
            timeline_penalty = 0.02 * (years_beyond / 10)
            honest_timeline_std = base_std + timeline_penalty
            
            upper_bound = future_preds + (1.96 * honest_timeline_std)
            lower_bound = future_preds - (1.96 * honest_timeline_std)
            
            hist_df = supreme_df.groupby('Year')['Temp_Anomaly'].mean().reset_index()
            
            fig_forecast = go.Figure()
            
            fig_forecast.add_trace(go.Scatter(
                x=hist_df['Year'], y=hist_df['Temp_Anomaly'], 
                mode='lines', name='Historical Reality', line=dict(color='royalblue', width=2)
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=future_years, y=upper_bound,
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=future_years, y=lower_bound,
                mode='lines', fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', 
                line=dict(width=0), name='95% Confidence Interval'
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=future_years, y=future_preds, 
                mode='lines', name=f'{country_name} Simulation', line=dict(color='red', width=4)
            ))
            
            fig_forecast.add_hline(
                y=1.5, line_dash="dash", line_color="orange", 
                annotation_text="1.5°C Danger Limit", annotation_position="bottom right"
            )

            fig_forecast.update_layout(
                title="Predictive Trajectory with Bayesian Uncertainty",
                xaxis_title="Year",
                yaxis_title="Temp Anomaly (°C)",
                hovermode="x unified"
            )

            st.info(
                "**Extrapolation Penalty**\n\n"
                "A specialized penalty is applied when features drift outside known historical bounds, "
                "forcing wider intervals to prevent false precision.", 
                icon="⚠️"
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
# =================================================
# EXPLAINABLE AI
# =================================================

st.markdown("---")
st.subheader(" What is driving the prediction?")

if not importance_df.empty:
    fig_imp = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        title="Feature Impact on Climate Prediction"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

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


cols = ['CO2_Emissions', 'Total_GHG', 'Average_Temperature']
ranking_scaler = MinMaxScaler()
ranking_df[cols] = ranking_scaler.fit_transform(ranking_df[cols])


ranking_df["risk_score"] = (
    ranking_df["CO2_Emissions"] * 0.4 +
    ranking_df["Total_GHG"] * 0.3 +
    ranking_df["Average_Temperature"] * 0.3
)


ranking_df["risk_score"] = (ranking_df["risk_score"] * 100).round(2)
ranking_df = ranking_df.sort_values("risk_score") 

st.dataframe(ranking_df.head(10))
# =================================================
# REGIONAL ANALYTICS
# =================================================

st.subheader(f"🌍 Regional Climate Overview")

region_summary = (
    supreme_df
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

st.plotly_chart(fig_region, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption("Intelligence provided by OWID, World Bank, and Supreme Neural Networks")
st.markdown("---")