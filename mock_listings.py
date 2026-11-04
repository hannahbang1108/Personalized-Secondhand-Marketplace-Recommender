"""
Temporary fake listings, so simulated buyers have something to compare.

This will be replaced by Mercari-based listings and later by real eBay data.
Any replacement just needs to return Listing objects from schema.py.

The numbers are picked to look roughly like eBay: most sellers have 98% to
100% positive feedback, a few sellers have far more reviews than everyone
else, and worse-condition items mention damage more often.
"""
import numpy as np
from schema import Listing, ITEMS_BY_CATEGORY, CONDITIONS, BRAND_TIERS


# Listings are scattered within 15 miles of this point. Change it to your area.
CENTER_LATITUDE, CENTER_LONGITUDE = 40.0, -75.0
RADIUS_MILES = 15

# How common each kind of listing is.
CATEGORY_SHARE = {"furniture": 0.35, "kitchen": 0.20, "electronics": 0.20,
                  "decor": 0.15, "bedding": 0.10}
CONDITION_SHARE = {"fair": 0.25, "good": 0.45, "like_new": 0.30}
BRAND_SHARE = {"unbranded": 0.35, "budget": 0.25, "mid": 0.28, "premium": 0.12}

# Chance that the description mentions damage, for each condition.
DAMAGE_CHANCE = {"fair": 0.70, "good": 0.30, "like_new": 0.05}

# Typical price for a good-condition, budget-brand item in each category.
TYPICAL_PRICE = {"furniture": 120, "kitchen": 35, "electronics": 90,
                 "decor": 40, "bedding": 30}
CONDITION_PRICE_FACTOR = {"fair": 0.55, "good": 0.8, "like_new": 1.0}
BRAND_PRICE_FACTOR = {"unbranded": 0.8, "budget": 0.9, "mid": 1.1, "premium": 1.6}


def make_listings(count, rng):
    return [make_listing(f"L{i:05d}", rng) for i in range(count)]


def make_listing(listing_id, rng):
    category = pick(CATEGORY_SHARE, rng)
    condition = pick(CONDITION_SHARE, rng)
    brand = pick(BRAND_SHARE, rng)
    item = str(rng.choice(ITEMS_BY_CATEGORY[category]))

    review_count = random_review_count(rng)
    # Sellers with lots of reviews are more often businesses, and businesses
    # more often accept returns.
    is_business = rng.random() < (0.6 if review_count > 1000 else 0.15)
    accepts_returns = rng.random() < (0.8 if is_business else 0.3)

    latitude, longitude = random_location(rng)
    title = item if brand == "unbranded" else f"{brand.capitalize()} {item}"

    return Listing(
        listing_id=listing_id,
        title=title,
        category=category,
        condition=condition,
        seller_feedback_percent=random_feedback_percent(rng),
        seller_review_count=review_count,
        business_seller=bool(is_business),
        returns_accepted=bool(accepts_returns),
        brand_tier=brand,
        mentions_damage=bool(rng.random() < DAMAGE_CHANCE[condition]),
        price=random_price(category, condition, brand, rng),
        latitude=latitude,
        longitude=longitude,
    )


def pick(shares, rng):
    """Pick one key from a {value: probability} dictionary."""
    return str(rng.choice(list(shares), p=list(shares.values())))


def random_feedback_percent(rng):
    # 90% of sellers are a little below 100% (a short drop-off). The other
    # 10% have a much wider spread, which creates the occasional bad seller.
    if rng.random() < 0.9:
        drop_below_100 = rng.exponential(0.8)
    else:
        drop_below_100 = rng.exponential(5.0)
    return float(round(max(100.0 - drop_below_100, 80.0), 1))


def random_review_count(rng):
    # A log-normal distribution: most sellers have dozens to hundreds of
    # reviews, and a few have tens of thousands.
    return int(rng.lognormal(mean=4.0, sigma=1.8))


def random_price(category, condition, brand, rng):
    price = (TYPICAL_PRICE[category]
             * CONDITION_PRICE_FACTOR[condition]
             * BRAND_PRICE_FACTOR[brand]
             * rng.lognormal(0, 0.5))   # random variation, mostly within +-50%
    return float(round(price, 2))


def random_location(rng):
    """A random point within RADIUS_MILES of the center."""
    # Taking the square root spreads points evenly over the circle's area,
    # instead of bunching them up near the center.
    distance = RADIUS_MILES * np.sqrt(rng.random())
    angle = rng.uniform(0, 2 * np.pi)

    # One degree of latitude is about 69 miles. Degrees of longitude get
    # narrower away from the equator, hence the cosine.
    miles_per_degree_longitude = 69.0 * np.cos(np.radians(CENTER_LATITUDE))
    latitude = CENTER_LATITUDE + distance * np.sin(angle) / 69.0
    longitude = CENTER_LONGITUDE + distance * np.cos(angle) / miles_per_degree_longitude
    return float(round(latitude, 5)), float(round(longitude, 5))