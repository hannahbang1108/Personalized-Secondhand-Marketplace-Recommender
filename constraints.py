import json
import urllib.parse
import urllib.request

import numpy as np

from schema import Listing


# Free, no API key, US addresses only
CENSUS_GEOCODER_URL = ("https://geocoding.geo.census.gov/geocoder/"
                       "locations/onelineaddress")

# returns (matched_address, lat, lon), or None if not found
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
    # x is longitude, y is latitude
    return (best["matchedAddress"], best["coordinates"]["y"],
            best["coordinates"]["x"])


def filter_listings(buyer, location, trip_miles, listings):
    # returns {item: listings that passed}
    wanted = {entry["item"] for entry in buyer.shopping_list}
    candidates = {item: [] for item in wanted}
    buyer_lat, buyer_lon = location

    for listing in listings:
        # Pythagoras in miles: 1 degree of latitude is ~69.09 miles, and degrees
        # of longitude shrink by cos(latitude) away from the equator
        dy = (listing.latitude - buyer_lat) * 69.09
        dx = (listing.longitude - buyer_lon) * 69.09 * np.cos(np.radians(buyer_lat))
        distance = np.hypot(dx, dy)

        # going there and back alone has to fit in the trip
        if (listing.item in wanted and listing.price <= buyer.budget
                and 2 * distance <= trip_miles):
            candidates[listing.item].append(listing)

    return candidates

def load_listings(path="data/listings.json"):
    with open(path) as f:
        return [Listing(**record) for record in json.load(f)]
