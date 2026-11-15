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
        # saved best_route answers, keyed by state
        "route_cache": {},
    }
