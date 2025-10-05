import requests
import pandas as pd
import numpy as np
import io
import urllib.parse as urlp
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings("ignore")

class WeatherDataCollector:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="petrichor_weather_backend")
        
    def get_time_series(self, start_date, end_date, latitude, longitude, variable):
        """
        Calls the NASA data service to get a time series
        """
        base_url = "https://hydro1.gesdisc.eosdis.nasa.gov/daac-bin/access/timeseries.cgi"
        query_parameters = {
            "variable": variable,
            "type": "asc2",
            "location": f"GEOM:POINT({longitude}, {latitude})",
            "startDate": start_date,
            "endDate": end_date,
        }
        full_url = base_url + "?" + \
            "&".join(["{}={}".format(key, urlp.quote(query_parameters[key])) for key in query_parameters])
        
        iteration = 0
        done = False
        while not done and iteration < 5:
            try:
                r = requests.get(full_url, timeout=30)
                if r.status_code == 200:
                    done = True
                else:
                    iteration += 1
            except requests.exceptions.RequestException:
                iteration += 1
        
        if not done:
            raise Exception(f"Error code {r.status_code} from url {full_url} : {r.text}")
        
        return r.text

    def parse_time_series(self, ts_str):
        """
        Parses the response from NASA data service.
        """
        lines = ts_str.split("\n")
        parameters = {}
        for line in lines[2:11]:
            if "=" in line:
                key, value = line.split("=", 1)
                parameters[key] = value
        
        df = pd.read_table(io.StringIO(ts_str), sep="\t",
                          names=["time", "data"],
                          header=10, parse_dates=["time"])
        return parameters, df

    def get_coordinates(self, address):
        """Get latitude and longitude from address"""
        location = self.geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            raise Exception(f"Location not found for: {address}")

    def collect_weather_data(self, start_date, end_date, address, years_back=10):
        """
        Collect comprehensive weather data for multiple years
        """
        # Parse dates
        split_start = start_date.split('-')
        start_year = int(split_start[0])
        start_month = split_start[1]
        start_day = split_start[2]

        split_end = end_date.split('-')
        end_year = int(split_end[0])
        end_month = split_end[1]
        end_day = split_end[2]

        # Get coordinates
        latitude, longitude = self.get_coordinates(address)

        # Variables to collect
        variables = {
            'Rainf': 'NLDAS2:NLDAS_FORA0125_H_v2.0:Rainf',
            'Qair': 'NLDAS2:NLDAS_FORA0125_H_v2.0:Qair',
            'Tair': 'NLDAS2:NLDAS_FORA0125_H_v2.0:Tair',
            'Wind_N': 'NLDAS2:NLDAS_FORA0125_H_v2.0:Wind_N',
            'Wind_E': 'NLDAS2:NLDAS_FORA0125_H_v2.0:Wind_E'
        }

        all_data = []

        for year_offset in range(1, years_back + 1):
            prev_year_start = f"{start_year - year_offset}-{start_month}-{start_day}"
            prev_year_end = f"{end_year - year_offset}-{end_month}-{end_day}"

            year_data = {'time': None}
            
            # Collect data for each variable
            for var_name, var_code in variables.items():
                try:
                    ts_data = self.get_time_series(prev_year_start, prev_year_end, 
                                                 latitude, longitude, var_code)
                    params, df = self.parse_time_series(ts_data)
                    
                    if year_data['time'] is None:
                        year_data['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    year_data[var_name] = df['data'].values
                    
                except Exception as e:
                    print(f"Error collecting {var_name} for year {start_year - year_offset}: {e}")
                    # Fill with zeros if data collection fails
                    if year_data['time'] is not None:
                        year_data[var_name] = np.zeros(len(year_data['time']))
                    else:
                        year_data[var_name] = np.array([0])

            # Create DataFrame for this year
            year_df = pd.DataFrame(year_data)
            
            # Add weather classification
            year_df['Weather'] = year_df['Rainf'].apply(self.classify_weather)
            
            all_data.append(year_df)

        # Combine all years
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Rename columns for consistency
        combined_df.rename(columns={
            'Qair': 'Humidity',
            'Tair': 'Air Temperature'
        }, inplace=True)

        return combined_df

    def classify_weather(self, rainfall):
        """Classify weather based on rainfall amount"""
        if rainfall == 0.0:
            return "No Rain"
        elif 0.0 < rainfall <= 0.25:
            return "Very Light Rain"
        elif 0.25 < rainfall <= 1.0:
            return "Light Rain"
        elif 1.0 < rainfall <= 4.0:
            return "Moderate Rain"
        elif 4.0 < rainfall <= 16:
            return "Heavy Rain"
        elif 16 < rainfall <= 50:
            return "Very Heavy Rain"
        else:
            return "Extreme Rain"