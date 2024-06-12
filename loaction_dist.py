import math
import requests


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance * 1000  # Convert to meters


def get_current_location():
    retry = 0
    while retry < 3:
        response = requests.get('https://api.ipbase.com/v1/json/')
        if response.status_code == 200:
            data = response.json()
            if 'latitude' in data and 'longitude' in data:
                latitude = data['latitude']
                longitude = data['longitude']
                return latitude, longitude
            else:
                return None, None
        elif response.status_code == 429:
            retry = retry + 1

    return 429, "Too Many Requests"


