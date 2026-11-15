import json
import math
import urllib.parse
import urllib.request

import numpy as np

from schema import Listing


# Free, no API key, US addresses only
CENSUS_GEOCODER_URL = ("https://geocoding.geo.census.gov/geocoder/"
                       "locations/onelineaddress")

# The farthest one-way distance (miles) a buyer can reasonably go to a pickup with each mode.
MODE_MAX_MILES = {
    "walk": 3,
    "transit": 10,
    "car": math.inf,
}


# Looks up an address and returns (matched_address, lat, lon), or None if it isn't found.
def geocode(address):
    query = urllib.parse.urlencode({"address": address,
                                    "benchmark": "Public_AR_Current",
                                    "format": "json"})
    with urllib.request.urlopen(f"{CENSUS_GEOCODER_URL}?{query}",
                                timeout=15) as response:
        matches = json.load(response)["result"]["addressMatches"]
    if not matches:
        return None
    best = matches[0]
    # The geocoder calls longitude x and latitude y.
    return (best["matchedAddress"], best["coordinates"]["y"],
            best["coordinates"]["x"])


# Returns the straight-line distance in miles between two points (or arrays of points).
def miles_between(lat1, lon1, lat2, lon2):
    # One degree of latitude is about 69.09 miles, and degrees of longitude
    # shrink by cos(latitude) away from the equator.
    dy = (lat2 - lat1) * 69.09
    dx = (lon2 - lon1) * 69.09 * np.cos(np.radians(lat1))
    return np.hypot(dx, dy)


# Returns {item: listings} with only the listings that pass every hard constraint.
def filter_listings(buyer, location, trip_miles, listings):
    candidates = {entry["item"]: [] for entry in buyer.shopping_list}

    # farthest buyer can go one way is set by the most generous mode they picked (e.g. walk + car -> car's limit).
    max_one_way = max((MODE_MAX_MILES[mode] for mode in buyer.transport), default=0)

    for listing in listings:
        distance = miles_between(*location, listing.latitude, listing.longitude)

        # Going there and back on its own has to fit within the trip limit, and the
        # one-way distance has to be within reach of their transport.
        if (listing.item in candidates and listing.price <= buyer.budget
                and 2 * distance <= trip_miles
                and distance <= max_one_way):
            candidates[listing.item].append(listing)

    return candidates


# Loads the listings saved by generate_mock_data.py.
def load_listings(path="data/listings.json"):
    with open(path) as f:
        return [Listing(**record) for record in json.load(f)]