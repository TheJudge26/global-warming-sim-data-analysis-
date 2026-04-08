# src/utils.py

# DONT TOUCH 
import pandas as pd
from src.config import CLEAN_DATA_PATH

def load_clean_data():
    
    return pd.read_csv(CLEAN_DATA_PATH)

def get_region_data(df, continent=None, climate_zone=None):
   
    filtered_df = df.copy()
    if continent:
        filtered_df = filtered_df[filtered_df['continent'] == continent]
    if climate_zone:
        filtered_df = filtered_df[filtered_df['climate_zone'] == climate_zone]
    return filtered_df