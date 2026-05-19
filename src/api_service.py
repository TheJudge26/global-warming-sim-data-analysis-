import requests
import pandas as pd
from datetime import datetime

def get_live_country_data(country_code="EGY"):
    """
    Fetches the most recent Population, GDP, and CO2 data from the World Bank API.
    country_code: ISO 3-letter country code (e.g., 'EGY' for Egypt, 'USA' for United States)
    """
    print(f"📡 Connecting to World Bank API for {country_code}...")
    
    
    indicators = {
        "Population": "SP.POP.TOTL",
        "GDP": "NY.GDP.MKTP.CD",
        "CO2_Emissions": "EN.ATM.CO2E.KT" 
    }
    
    live_data = {'Year': datetime.now().year}
    
    for feature_name, indicator_code in indicators.items():
        
        url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json&mrnev=1"
        
        try:
            response = requests.get(url)
            response.raise_for_status() 
            data = response.json()
            
            
            if len(data) > 1 and data[1]:
                value = data[1][0]['value']
                live_data[feature_name] = value
            else:
                live_data[feature_name] = 0.0
                print(f"⚠️ Warning: No data found for {feature_name}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error for {feature_name}: {e}")
            live_data[feature_name] = 0.0

    
    df_live = pd.DataFrame([live_data])
    print(" Live Data Acquired successfully!\n")
    
    return df_live


if __name__ == "__main__":
    
    egypt_current_data = get_live_country_data("EGY")
    print("Current Live Data for Egypt:")
    print(egypt_current_data)