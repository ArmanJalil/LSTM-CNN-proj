import pandas as pd
import requests
import time
from datetime import datetime

# Step 1: Set API parameters
OWM_API_KEY = "f52e3e4141fcc8830b2540ab27471ed1"
CENTER_LAT = 32.661697
CENTER_LON = 51.672492
POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Time range
WEATHER_START = "2020-01-01"
POLLUTION_START = int(datetime(2020, 11, 27).timestamp())  # OpenWeatherMap limit
END_TIME = int(datetime(2025, 4, 18, 23, 59).timestamp())
END_DATE = "2025-04-18"

# Step 2: Generate 3x3 grid points within 15 km (~5 km spacing)
def get_grid_points(center_lat, center_lon, radius_km=15, steps=3):
    points = []
    # Approximate degrees per km: 1 degree ~ 111 km
    deg_per_km = 1 / 111
    step_size = (radius_km / 2) * deg_per_km  # ~0.0675° for 7.5 km
    for i in range(-steps//2 + 1, steps//2 + 1):
        for j in range(-steps//2 + 1, steps//2 + 1):
            lat = center_lat + i * step_size
            lon = center_lon + j * step_size * (1 / abs(math.cos(math.radians(center_lat))))
            points.append((lat, lon))
    return points

import math
grid_points = get_grid_points(CENTER_LAT, CENTER_LON, radius_km=15, steps=3)
print(f"Fetching data for {len(grid_points)} points: {grid_points}")

# Step 3: Fetch pollution data (OpenWeatherMap)
def fetch_pollution(lat, lon, start, end):
    params = {
        "lat": lat,
        "lon": lon,
        "start": start,
        "end": end,
        "appid": OWM_API_KEY
    }
    response = requests.get(POLLUTION_URL, params=params)
    if response.status_code == 200:
        data = response.json().get('list', [])
        print(f"Fetched {len(data)} pollution records for ({lat}, {lon})")
        return data
    print(f"Pollution API error at ({lat}, {lon}): {response.status_code}, {response.text}")
    return []

# Step 4: Fetch weather data (Open-Meteo)
def fetch_weather(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation,wind_speed_10m",
        "timezone": "auto"
    }
    response = requests.get(METEO_URL, params=params)
    if response.status_code == 200:
        return response.json().get('hourly', {})
    print(f"Weather API error at ({lat}, {lon}): {response.status_code}, {response.text}")
    return {}

# Step 5: Fetch data for all points
pollution_records = []
weather_records = []
chunk_size = 5 * 24 * 3600  # 5 days

for lat, lon in grid_points:
    # Pollution data
    current_start = POLLUTION_START
    while current_start < END_TIME:
        current_end = min(current_start + chunk_size, END_TIME)
        data = fetch_pollution(lat, lon, current_start, current_end)
        for entry in data:
            pollution_records.append({
                'time': pd.to_datetime(entry['dt'], unit='s'),
                'latitude': lat,
                'longitude': lon,
                'OpenWeather_CO': entry['components']['co'],
                'OpenWeather_NO': entry['components']['no'],
                'OpenWeather_NO2': entry['components']['no2'],
                'OpenWeather_O3': entry['components']['o3'],
                'OpenWeather_SO2': entry['components']['so2'],
                'OpenWeather_PM2_5': entry['components']['pm2_5'],
                'OpenWeather_PM10': entry['components']['pm10'],
                'OpenWeather_NH3': entry['components']['nh3']
            })
        current_start = current_end
        time.sleep(1)

    # Weather data
    weather_response = fetch_weather(lat, lon, WEATHER_START, END_DATE)
    times = pd.to_datetime(weather_response.get('time', []))
    precip = weather_response.get('precipitation', [])
    wind_speed = weather_response.get('wind_speed_10m', [])
    for t, p, w in zip(times, precip, wind_speed):
        weather_records.append({
            'time': t,
            'latitude': lat,
            'longitude': lon,
            'OpenMeteo_Precipitation': p,
            'OpenMeteo_WindSpeed': w
        })

# Step 6: Create DataFrames
df_pollution = pd.DataFrame(pollution_records)
df_weather = pd.DataFrame(weather_records)

# Step 7: Merge data
if not df_pollution.empty and not df_weather.empty:
    df_pollution['time'] = df_pollution['time'].dt.round('h')
    df_weather['time'] = df_weather['time'].dt.round('h')
    df_combined = pd.merge(df_pollution, df_weather, on=['time', 'latitude', 'longitude'], how='outer')
else:
    print("Error: One or both DataFrames are empty")
    print("Pollution DataFrame:", df_pollution.head())
    print("Weather DataFrame:", df_weather.head())
    df_combined = pd.DataFrame()

# Step 8: Save to CSV
df_combined.to_csv('Esfahan_Pollution_Weather_15km_20200101_20250418.csv', index=False)
print("Data saved as 'Esfahan_Pollution_Weather_15km_20200101_20250418.csv'")