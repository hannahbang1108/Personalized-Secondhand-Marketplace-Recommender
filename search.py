import itertools
import math
import random

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
        "price_weight": 1.0,
        "trip_weight": 1.0,
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
    return (-average_match
            + problem["price_weight"] * price_share
            + problem["trip_weight"] * trip_share)

# Returns a copy of the state with one item's listing swapped for a different listing of that item.
def get_random_neighbor(problem, state):
    # only items with more than one listing can change
    items_with_choices = [item_num for item_num, listings in enumerate(problem["listings_by_item"])
                          if len(listings) > 1]
    if not items_with_choices:
        return state
    item_num = random.choice(items_with_choices)
    # pick from one fewer than all the listings, then shift past the current one,
    # so the new listing is always different
    new_listing_num = random.randrange(len(problem["listings_by_item"][item_num]) - 1)
    new_listing_num += new_listing_num >= state[item_num]
    return state[:item_num] + (new_listing_num,) + state[item_num + 1:]

# First-choice hill climbing on limit_violation: starts from the cheapest listing for each item
# and keeps a random swap only if it reduces how far over the budget and trip limit the state is.
# Returns a state within both limits, or None if it can't find one.
def first_choice_hill_climbing(problem, max_steps=2000):
    current = tuple(min(range(len(listings)), key=lambda i: listings[i].price)
                    for listings in problem["listings_by_item"])
    steps = 0
    while limit_violation(problem, current) > 0 and steps < max_steps:
        neighbor = get_random_neighbor(problem, current)
        # keep whichever of the two is less over the limits
        current = min([current, neighbor], key=lambda state: limit_violation(problem, state))
        steps += 1
    return current if limit_violation(problem, current) == 0 else None

# Simulated annealing - returns the best state found.
def simulated_annealing(problem, s, T, d, min_temp=1e-4):
    current = s
    best = s
    # stops once it has cooled down
    while T > min_temp:
        neighbor = get_random_neighbor(problem, current)
        current_energy = energy(problem, current)
        neighbor_energy = energy(problem, neighbor)
        # a neighbor over the budget or trip limit is never taken
        if limit_violation(problem, neighbor) == 0:
            # better or equal neighbors are always taken
            if neighbor_energy <= current_energy:
                current = neighbor
            # worse neighbors are taken with probability e^((current - neighbor) / T)
            elif random.random() <= math.exp((current_energy - neighbor_energy) / T):
                current = neighbor
        # best only changes when the current state beats it
        if energy(problem, current) < energy(problem, best):
            best = current
        T = T * d
    return best

# Returns the best state found, or None if no state fits within the budget and trip limit.
def solve(problem, d=0.9998):
    start = first_choice_hill_climbing(problem)
    if start is None:
        return None
    # try 100 random moves from the start and see how much worse the bad ones are
    start_energy = energy(problem, start)
    deltas = [energy(problem, get_random_neighbor(problem, start)) - start_energy for _ in range(100)]
    worse_deltas = [delta for delta in deltas if delta > 0]
    # pick T where about 80% of worse moves are accepted at first: e^(-delta / T) = 0.8
    T = -(sum(worse_deltas) / len(worse_deltas)) / math.log(0.8) if worse_deltas else 1.0
    return simulated_annealing(problem, start, T, d)