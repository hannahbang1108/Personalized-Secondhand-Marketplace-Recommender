import itertools
import math

import numpy as np

from constraints import miles_between
from mock_buyers import make_true_weights, true_score


# Sets up the search problem of picking one listing for each item on the shopping list.
# A state is a tuple with one listing_num per item, e.g. (3, 0, 12) = 4th desk, 1st sofa, 13th tv.
def make_problem(buyer, home, trip_limit, candidates):
    weights = make_true_weights(buyer)
    # listings_by_item[item_num][listing_num] is a Listing
    listings_by_item = list(candidates.values())
    return {
        "buyer": buyer,
        "home": home,
        "trip_limit": trip_limit,
        "item_names": list(candidates),
        "listings_by_item": listings_by_item,
        # scores each listing based on the buyer's preferences, scored once so the step 3 model never reruns during the search
        "match_scores": [[true_score(weights, listing) for listing in listings] for listings in listings_by_item], 
        "prices": [[listing.price for listing in listings] for listings in listings_by_item],  # same shape as listings_by_item
        # saved shortest_distance answers, keyed by state
        "route_cache": {},
    }

# Returns (miles, pickup order) for the shortest loop from home to every pickup and back.
# The order is a tuple of item_nums, e.g. (2, 0, 1) = tv first, then desk, then sofa.
def shortest_distance(problem, state):
    cache = problem["route_cache"]
    # if we've already worked out this state, return the saved answer
    if state in cache:
        return cache[state]

    # the Listing this state picks for each item
    chosen_listings = [problem["listings_by_item"][item_num][listing_num]
                       for item_num, listing_num in enumerate(state)]

    shortest_miles = math.inf
    shortest_order = None
    # try every order to pick up the items and keep the shortest loop
    for order in itertools.permutations(range(len(state))):
        # home, then each pickup in this order, then home again
        stops = [problem["home"]]
        stops += [(chosen_listings[item_num].latitude, chosen_listings[item_num].longitude) for item_num in order]
        stops += [problem["home"]]

        # add up the miles of each leg, from one stop to the next
        miles = sum(miles_between(*start, *end) for start, end in zip(stops, stops[1:]))

        if miles < shortest_miles:
            shortest_miles = miles
            shortest_order = order

    cache[state] = (float(shortest_miles), shortest_order)
    return cache[state]

# Returns how far over the budget and trip limit the state is (0 if it is within both).
def limit_violation(problem, state):
    budget = problem["buyer"].budget
    total_price = sum(problem["prices"][item_num][listing_num] for item_num, listing_num in enumerate(state))
    over_budget = max(0, total_price - budget)
    over_trip = max(0, shortest_distance(problem, state)[0] - problem["trip_limit"])
    # divided by each limit so dollars and miles can be added together
    return over_budget / budget + over_trip / problem["trip_limit"]

# Returns the energy to minimize: low means good matches, low price, and a short trip.
def energy(problem, state):
    # higher is better
    average_match = sum(problem["match_scores"][item_num][listing_num]
                        for item_num, listing_num in enumerate(state)) / len(state)
    # total price as a share of the budget
    price_share = sum(problem["prices"][item_num][listing_num]
                      for item_num, listing_num in enumerate(state)) / problem["buyer"].budget
    # trip miles as a share of the trip limit
    trip_share = shortest_distance(problem, state)[0] / problem["trip_limit"]
    # match is subtracted so a higher match lowers the energy
    return -average_match + price_share + trip_share