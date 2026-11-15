"""
Simulated buyers for training and evaluating the matching model.

We don't have real users yet, so we make up hundreds of buyers. For each one:

  1. make_buyer()           fills out the onboarding form: budget, transport,
                            shopping list, and priority ratings
  2. make_true_weights()    decides how much the buyer really cares about each
                            listing feature, based on their priority ratings
  3. simulate_choices()     has the buyer rate listings from 1 to 5, using
                            those true weights

The true weights are the answer key. The matching model never sees them; we
only use them afterwards to check how well the model learned each buyer.
"""
import numpy as np

from mock_listings import random_location
from schema import (Buyer, ITEMS_BY_CATEGORY, TRANSPORT_MODES, PRIORITIES,
                    LISTING_FEATURE_NAMES, listing_to_features)


FIRST_NAMES = ["Maya", "Jordan", "Priya", "Luis", "Chen", "Amara", "Sam", "Elena",
               "Omar", "Grace", "Diego", "Hana", "Noah", "Fatima", "Theo", "Ines",
               "Kwame", "Lena", "Ravi", "Zoe"]

# How likely a buyer is to want items from each category.
CATEGORY_POPULARITY = {"furniture": 0.40, "kitchen": 0.20, "electronics": 0.15,
                       "decor": 0.15, "bedding": 0.10}

# Chance that a buyer is willing to use each transport mode.
TRANSPORT_WILLINGNESS = {"walk": 0.70, "transit": 0.50, "car": 0.45}


# ---------------------------------------------------------------------------
# Step 1: filling out the onboarding form
# ---------------------------------------------------------------------------

def make_buyer(rng, buyer_id, home_rng):
    """Create one buyer with randomly chosen form answers.

    The home and trip limit come from their own random stream (home_rng), so
    adding them didn't change the rest of the generated data for a given seed.
    """
    home_latitude, home_longitude = random_location(home_rng)   # same area as the listings
    return Buyer(
        buyer_id=buyer_id,
        name=str(rng.choice(FIRST_NAMES)),
        priorities={p: int(rng.integers(1, 6)) for p in PRIORITIES},   # 1 to 5
        transport=pick_transport_modes(rng),
        budget=float(round(rng.uniform(150, 2000), -1)),   # rounded to $10
        shopping_list=pick_shopping_list(rng),
        home_latitude=home_latitude,
        home_longitude=home_longitude,
        trip_miles=float(home_rng.integers(10, 41)),   # 10 to 40 miles
    )


def pick_transport_modes(rng):
    modes = [m for m in TRANSPORT_MODES if rng.random() < TRANSPORT_WILLINGNESS[m]]
    if not modes:   # everyone needs at least one way to get around
        modes = [str(rng.choice(TRANSPORT_MODES))]
    return modes


def pick_shopping_list(rng):
    """Pick 2 to 5 different items, favoring popular categories."""
    all_items = [(item, category)
                 for category, items in ITEMS_BY_CATEGORY.items()
                 for item in items]

    # Split each category's popularity evenly across its items, so a category
    # with many items isn't picked more often just because of its size.
    item_odds = np.array([CATEGORY_POPULARITY[category] / len(ITEMS_BY_CATEGORY[category])
                          for _, category in all_items])
    item_odds /= item_odds.sum()

    num_items = int(rng.integers(2, 6))
    picks = rng.choice(len(all_items), size=num_items, replace=False, p=item_odds)
    return [{"item": all_items[i][0], "category": all_items[i][1]} for i in picks]


# ---------------------------------------------------------------------------
# Step 2: deciding how much each buyer truly cares about each feature
# ---------------------------------------------------------------------------

# For each listing feature: which priority controls it, and how much it
# matters to a buyer who gave that priority an average rating of 3.
# Positive means the buyer likes it; negative means they dislike it.
#
# A buyer's actual weight is this base weight scaled by their rating:
# a rating of 5 makes it 5/3 as strong, a rating of 1 makes it 1/3 as strong.
BASE_WEIGHTS = {
    "condition":                ("condition",    1.0),
    "mentions_damage":          ("condition",   -0.5),
    "seller_negative_feedback": ("seller_trust", -0.6),
    "seller_review_count":      ("seller_trust", 0.8),
    "business_seller":          ("seller_trust", 0.2),
    "returns_accepted":         ("seller_trust", 0.2),
    "brand_budget":             ("brand",        0.1),
    "brand_mid":                ("brand",        0.4),
    "brand_premium":            ("brand",        0.8),
}
assert set(BASE_WEIGHTS) == set(LISTING_FEATURE_NAMES), "every feature needs a weight"

# Some preferences depend on two features at once, which a plain weighted sum
# can't capture. Each entry is (feature A, feature B, priority, base weight):
# the buyer's score changes by weight * A * B, so the effect only appears when
# both features are present. Like BASE_WEIGHTS, the weight is scaled by the
# buyer's rating for that priority. The matching model never sees these; it
# has to discover them from the ratings.
INTERACTIONS = [
    # Paying for a premium brand, buyers expect it in good shape.
    ("condition", "brand_premium", "condition", 2.0),
    # Lots of reviews mean more coming from a business seller.
    ("seller_review_count", "business_seller", "seller_trust", 3.0),
]


def make_true_weights(buyer):
    """Return {feature name: weight} describing what this buyer really wants."""
    true_weights = {}
    for feature in LISTING_FEATURE_NAMES:
        priority, base_weight = BASE_WEIGHTS[feature]
        rating = buyer.priorities[priority]
        true_weights[feature] = base_weight * rating / 3.0
    for feature_a, feature_b, priority, base_weight in INTERACTIONS:
        rating = buyer.priorities[priority]
        true_weights[f"{feature_a} x {feature_b}"] = base_weight * rating / 3.0
    return true_weights


def true_score(true_weights, listing):
    """How much the buyer really likes a listing: a weighted sum of its features,
    plus the interaction effects."""
    weights = np.array([true_weights[f] for f in LISTING_FEATURE_NAMES])
    features = listing_to_features(listing)
    score = weights @ features
    for feature_a, feature_b, _, _ in INTERACTIONS:
        product = (features[LISTING_FEATURE_NAMES.index(feature_a)]
                   * features[LISTING_FEATURE_NAMES.index(feature_b)])
        score += true_weights[f"{feature_a} x {feature_b}"] * product
    return float(score)


# ---------------------------------------------------------------------------
# Step 3: simulating ratings from 1 to 5
# ---------------------------------------------------------------------------

# Share of ratings at each star level, from 1 to 5. Real ratings lean
# positive, so these do too. To be calibrated against real Amazon ratings.
RATING_SHARE = [0.08, 0.12, 0.22, 0.30, 0.28]


def simulate_choices(buyers, true_weights_by_buyer, listings, rng,
                     questions_per_buyer=40, consistency=1.5):
    """Have each buyer rate listings from 1 (hate it) to 5 (love it).

    Buyers aren't perfectly consistent, just like real people, so each rating
    starts from the buyer's true score plus random noise:

        noisy score = true score + (a random number around 0) / consistency

    The noisy scores are then turned into 1 to 5 stars so that the share of
    each star level matches RATING_SHARE. A higher consistency means ratings
    follow the true preferences more closely.
    """
    listings_by_category = {}
    for listing in listings:
        listings_by_category.setdefault(listing.category, []).append(listing)

    choices = []
    for buyer in buyers:
        weights = true_weights_by_buyer[buyer.buyer_id]
        wanted_categories = [entry["category"] for entry in buyer.shopping_list]
        already_rated = set()

        for _ in range(questions_per_buyer):
            # Mostly ask about categories the buyer is shopping for, but
            # sometimes others, so the model sees a range of listings.
            if rng.random() < 0.8:
                category = rng.choice(wanted_categories)
            else:
                category = rng.choice(list(listings_by_category))

            options = listings_by_category[category]
            # The same buyer never rates the same listing twice.
            listing = options[rng.integers(len(options))]
            while listing.listing_id in already_rated:
                listing = options[rng.integers(len(options))]
            already_rated.add(listing.listing_id)

            score = true_score(weights, listing)
            choices.append({
                "buyer_id": buyer.buyer_id,
                "listing_id": listing.listing_id,
                "noisy_score": score + rng.standard_normal() / consistency,
                # Only for evaluation: the model must never train on this.
                "true_score": round(score, 4),
            })

    # Turn noisy scores into stars: the lowest 8% become 1 star, the next 12%
    # become 2 stars, and so on, following RATING_SHARE.
    cutoffs = np.quantile([c["noisy_score"] for c in choices],
                          np.cumsum(RATING_SHARE)[:-1])
    for c in choices:
        noisy_score = c.pop("noisy_score")
        c["rating"] = int(np.searchsorted(cutoffs, noisy_score) + 1)
    return choices
