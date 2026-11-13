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
# to only those that are feasible for that buyer. Finally, print a report of how many
# listings were removed for each reason, and how many remain for each item.
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
    # returns {item: listings that passed} and {reason: number removed}
    wanted = {entry["item"] for entry in buyer.shopping_list}
    removed = {"not on shopping list": 0, "over budget": 0,
               "too far away": 0, "leaves too little for other items": 0}
    candidates = {item: [] for item in wanted}
    buyer_lat, buyer_lon = location

    for listing in listings:
        # Pythagoras in miles: 1 degree of latitude is ~69 miles, and degrees
        # of longitude shrink by cos(latitude) away from the equator
        dy = (listing.latitude - buyer_lat) * 69.0
        dx = (listing.longitude - buyer_lon) * 69.0 * np.cos(np.radians(buyer_lat))
        distance = np.hypot(dx, dy)

        item = listing.item
        if item not in wanted:
            removed["not on shopping list"] += 1
        elif listing.price > buyer.budget:
            removed["over budget"] += 1
        elif distance > max_miles:
            removed["too far away"] += 1
        else:
            candidates[item].append(listing)

    # drop listings that wouldn't leave enough money for the cheapest
    # option of every other item
    if all(candidates.values()):
        cheapest = {item: min(l.price for l in options)
                    for item, options in candidates.items()}
        total_cheapest = sum(cheapest.values())
        for item, options in candidates.items():
            money_for_others = buyer.budget - (total_cheapest - cheapest[item])
            kept = [l for l in options if l.price <= money_for_others]
            removed["leaves too little for other items"] += len(options) - len(kept)
            candidates[item] = kept

    return candidates, removed


def check_feasible(buyer, candidates):
    # returns a message if nothing can work, otherwise None
    missing = [item for item, options in candidates.items() if not options]
    if missing:
        return (f"No listings left for: {', '.join(missing)}. "
                "Try raising your max distance or your budget.")
    total_cheapest = sum(min(l.price for l in opts) for opts in candidates.values())
    if total_cheapest > buyer.budget:
        return (f"The cheapest option for every item adds up to "
                f"${total_cheapest:.2f}, which is over your ${buyer.budget:.2f} budget.")
    return None


# --- Try it out ---

def load_listings(path="data/listings.json"):
    with open(path) as f:
        return [Listing(**record) for record in json.load(f)]


def print_report(buyer, max_miles, candidates, removed):
    print(f"\n=== Results for {buyer.name} ===")
    print(f"Pickup range: {max_miles:g} miles")
    for reason, count in removed.items():
        print(f"  removed {count:5d}  {reason}")
    for item, options in candidates.items():
        print(f"  {item:15s} {len(options):4d} listings left")
    problem = check_feasible(buyer, candidates)
    print("\n" + (problem or "At least one combination fits your budget."))


if __name__ == "__main__":
    buyer, location, max_miles = ask_buyer()
    candidates, removed = filter_listings(buyer, location, max_miles,
                                          load_listings())
    print_report(buyer, max_miles, candidates, removed)
