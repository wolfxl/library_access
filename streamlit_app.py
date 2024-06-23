import streamlit as st
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import requests
import numpy as np
from geopy.distance import geodesic
import pandas as pd
from scipy.stats import percentileofscore
from dotenv import load_dotenv, find_dotenv
import os
import geopandas as gpd
import folium
from streamlit_folium import st_folium

load_dotenv(find_dotenv())
api_key = os.getenv('MAPBOX_ACCESS_TOKEN')


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_library_data(shapefile_path):
    return gpd.read_file(shapefile_path)


@st.cache_data
def geocode_address(address):
    geolocator = Nominatim(user_agent="library_locator")
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            return None
    except GeocoderTimedOut:
        return None

@st.cache_data
def get_distance_from_mapbox(origin, destination, access_token):
    url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
    params = {
        "access_token": api_key,
        "geometries": "geojson",
    }
    response = requests.get(url, params=params)
    data = response.json()
    if response.status_code == 200 and data['routes']:
        distance = data['routes'][0]['distance'] / 1000  # convert meters to kilometers
        return distance
    return None

def calculate_euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def find_closest_library(address, libraries):
  
    # Calculate distances to each library
    distances = libraries.apply(lambda row: calculate_euclidean_distance(coordinates, (row.geometry.y, row.geometry.x)), axis=1)
    
    # Find the closest library
    closest_idx = distances.idxmin()
    closest_library = libraries.loc[closest_idx]
    
    return closest_library.geometry.y, closest_library.geometry.x

@st.cache_data
def dist_list(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if the 'dist' column exists
    if 'dist' not in df.columns:
        raise ValueError("The CSV file does not contain a 'dist' column.")
    
    distances = df['dist']
    
    return distances

def is_within_boundary(lat, lon, north=33.3, south=32.5, east=-96.5, west=-97.5):

    return south <= lat <= north and west <= lon <= east


def calc_dist(my_route):
    length_in_miles = 0
    for i in range(len(my_route) - 1):
        start = (my_route[i][1], my_route[i][0])
        end = (my_route[i + 1][1], my_route[i + 1][0])
        length_in_miles += geodesic(start, end).miles
    return length_in_miles

def find_route(start_coords, end_coords):

    # Step 2: Use the Mapbox Directions API to get the route
    directions_url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
    params = {
        'access_token': api_key,
        'geometries': 'geojson'
    }
    response = requests.get(directions_url, params=params)
    data = response.json()

    # Extract the route geometry
    route = data['routes'][0]['geometry']['coordinates']
    return route



if "old_address" not in st.session_state:
    st.session_state.old_address = None

if "old_coor" not in st.session_state:
    st.session_state.old_coor = None

if "old_closest" not in st.session_state:
    st.session_state.old_closest = None

if "old_route" not in st.session_state:
    st.session_state.old_route = None

if "error_address" not in st.session_state:
    st.session_state.error_address = None

st.set_page_config(layout="wide")


load_css("styles.css")
address = st.text_input(label='Enter the address of interest to calculate the distance from your closest library:', key="address_input")


shapefile_path = 'data/lib_subset.shp'
libraries = load_library_data(shapefile_path)


file_path = 'data/distance2.csv'
distances =dist_list(file_path)
percentile = 0


lower_bound = np.percentile(distances, 0)
upper_bound = np.percentile(distances, 98)
distances = [d for d in distances if lower_bound <= d <= upper_bound]


# Set the center of the map to DFW area
map_center = [32.7767, -96.7970]  # Dallas, TX coordinates
my_dist=0

map_dfw = folium.Map(location=map_center, zoom_start=10, tiles='CartoDB Positron')



for idx, row in libraries.iterrows():
    popup_content = f'<div style="width:200px; font-size:12px;">{row["names"]}</div>'
    popup = folium.Popup(popup_content, max_width=250)
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=3,  # Adjust the size of the marker here
        popup=popup,
        color='red',
        fill=True,
        fill_opacity=0.5
    ).add_to(map_dfw)

if address:
    if st.session_state.old_address==address and st.session_state.error_address==False:
        coordinates=st.session_state.old_coor
        if coordinates:
            folium.Marker(
                location=coordinates,
                popup=f'<div style="font-size:16px;">{address}</div>',
                icon=folium.Icon(color='blue')
            ).add_to(map_dfw)
        print('old',coordinates)


        my_route=st.session_state.old_route

        if my_route:
            polyline=folium.PolyLine(
            locations=[(coord[1], coord[0]) for coord in my_route],
            color='blue',
            weight=5,
            opacity=0.7
            ).add_to(map_dfw)
            map_dfw.fit_bounds(polyline.get_bounds())
            my_dist=calc_dist(my_route)
            percentile =  100- percentileofscore(distances, my_dist)
            #percentile = np.percentile(distances, np.sum(distances <= my_dist) / len(distances) * 100)
            print('dist:',my_dist)
            



    else:
        st.session_state.old_address=address
        
        coordinates = geocode_address(address)
        if isinstance(coordinates, tuple) and len(coordinates) == 2:
            if is_within_boundary(coordinates[0], coordinates[1]) :
                st.session_state.error_address=False
                print('I am here!!')
                st.session_state.old_coor=coordinates
                if coordinates:
                    folium.Marker(
                        location=coordinates,
                        popup=f'<div style="font-size:16px;">{address}</div>',
                        icon=folium.Icon(color='blue')
                    ).add_to(map_dfw)
                print('new',coordinates)

                result=find_closest_library(coordinates, libraries)
                st.session_state.old_closest=result
                my_route=find_route(coordinates, result)
                st.session_state.old_route=my_route
                if my_route:
                    polyline =folium.PolyLine(
                    locations=[(coord[1], coord[0]) for coord in my_route],
                    color='blue',
                    weight=5,
                    opacity=0.7
                ).add_to(map_dfw)
                #map_dfw.fit_bounds(polyline.get_bounds())
            else:
                st.warning("The address could not be correctly located, please enter a new address.")
                st.session_state.error_address=True


        else:
            st.warning("The address could not be correctly located, please enter a new address.")
            st.session_state.error_address=True


   

# Display the map
st_folium(map_dfw, use_container_width=True, height=500)



st.write(f"The distance between your home and the closest library is {my_dist:.1f} miles, the distance beats {percentile:.2f}% of the DFW residents ")


