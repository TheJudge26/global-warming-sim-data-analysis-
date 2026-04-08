import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#DONT CHANGE THE PATH
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "global_warming_dataset.csv")
CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "global_warming_cleaned.csv")


CONTINENTS = [
    "Africa", "Asia", "Europe", "North America", 
    "South America", "Oceania", "Antarctica"
]

CLIMATE_ZONES = [
    "Tropical", "Subtropical", "Temperate", "Polar"
]


WEATHER_COLS = ["Average_Temperature", "Average_Rainfall", "Extreme_Weather_Events"]
POLLUTION_COLS = ["CO2_Emissions", "Methane_Emissions", "Air_Pollution_Index"]