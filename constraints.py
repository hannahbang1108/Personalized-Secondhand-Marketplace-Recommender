import json
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from schema import (Buyer, Listing, ITEMS_BY_CATEGORY, TRANSPORT_MODES,
                    PRIORITIES)


# Free, no API key, US addresses only
CENSUS_GEOCODER_URL = ("https://geocoding.geo.census.gov/geocoder/"
                       "locations/onelineaddress")

ALL_ITEMS = {item: category
             for category, items in ITEMS_BY_CATEGORY.items()
             for item in items}


# Create a Buyer profile by asking questions in the console. Then filter a list of listings
# to only those that are feasible for that buyer. Finally, print how many
# listings remain for each item.
def ask_buyer(buyer_id="U_live"):
    print("=== Secondhand Marketplace Recommender: onboarding ===\n")
    name = input("Your name: ").strip()
    while not name:
        name = input("This is required. Your name: ").strip()

    budget = ask_number("Total budget for everything, in dollars: ")

    print(f"\nTransport options: {', '.join(TRANSPORT_MODES)}")
    transport = ask_choices("Which modes of transportation can you use? (comma-separated): ",
                            TRANSPORT_MODES)

    print("\nYour address is used to measure distance to each pickup.")
    while True:
        address = input("Street address, city, state (e.g. 360 Huntington Ave, Boston, MA): ").strip()
        if not address:
            print("An address is required.")
            continue
        try:
            match = geocode(address)
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            print(f"Couldn't reach the geocoder ({error}). Please try again.")
            continue
        if match is None:
            print("Couldn't find that address. Check the spelling and include the city and state.")
            continue
        matched_address, latitude, longitude = match
        print(f"Found: {matched_address}")
        location = (latitude, longitude)
        break

    max_miles = ask_number("Farthest you're willing to go for pickup, in miles: ",
                           minimum=0.1)

    print("\nItems you can shop for:")
    for category, items in ITEMS_BY_CATEGORY.items():
        print(f"  {category:12s} {', '.join(items)}")
    items = ask_choices("What's on your shopping list? (comma-separated): ",
                        list(ALL_ITEMS))
    shopping_list = [{"item": i, "category": ALL_ITEMS[i]} for i in items]

    print("\nHow much does each of these matter to you? 1 = doesn't matter, 5 = matters a lot")
    priorities = {p: int(ask_number(f"  {p.replace('_', ' ')}: ",
                                    minimum=1, maximum=5, whole=True))
                  for p in PRIORITIES}

    buyer = Buyer(buyer_id=buyer_id, name=name, priorities=priorities,
                  transport=transport, budget=budget,
                  shopping_list=shopping_list)
    return buyer, location, max_miles


def ask_number(prompt, minimum=None, maximum=None, whole=False):
    while True:
        answer = input(prompt).strip().replace("$", "").replace(",", "")
        try:
            value = int(answer) if whole else float(answer)
        except ValueError:
            print("Please enter a number.")
            continue
        if minimum is not None and value < minimum:
            print(f"Please enter at least {minimum}.")
        elif maximum is not None and value > maximum:
            print(f"Please enter at most {maximum}.")
        else:
            return value


def ask_choices(prompt, allowed):
    while True:
        answers = [a.strip().lower() for a in input(prompt).split(",") if a.strip()]
        unknown = [a for a in answers if a not in allowed]
        if not answers:
            print("Please pick at least one.")
        elif unknown:
            print(f"Not recognized: {', '.join(unknown)}")
        else:
            return list(dict.fromkeys(answers))  # removes duplicates

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


def filter_listings(buyer, location, max_miles, listings):
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

        if (listing.item in wanted and listing.price <= buyer.budget
                and distance <= max_miles):
            candidates[listing.item].append(listing)

    return candidates

def load_listings(path="data/listings.json"):
    with open(path) as f:
        return [Listing(**record) for record in json.load(f)]


def print_report(buyer, max_miles, candidates):
    print(f"\n=== Results for {buyer.name} ===")
    print(f"Pickup range: {max_miles:g} miles")
    for item, options in candidates.items():
        print(f"  {item:15s} {len(options):4d} listings left")

    # the search needs at least one listing for every item
    missing = [item for item, options in candidates.items() if not options]
    if missing:
        print(f"\nNo listings left for: {', '.join(missing)}. "
              "Try raising your max distance or your budget.")
    else:
        print("\nEvery item has at least one listing left.")


if __name__ == "__main__":
    buyer, location, max_miles = ask_buyer()
    candidates = filter_listings(buyer, location, max_miles, load_listings())
    print_report(buyer, max_miles, candidates)
