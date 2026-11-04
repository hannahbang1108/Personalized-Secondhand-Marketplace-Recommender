"""
TEMPORARY listing generator so simulated buyers have something to compare.
Replace with the Mercari-grounded generator (and later the eBay adapter);
both must return the same Listing objects from schema.py.

Distributions are chosen to look roughly like eBay: most sellers sit at
98-100% positive feedback, feedback scores are heavy-tailed, and keywords
depend on condition.
"""
import numpy as np
from schema import Listing, CATEGORIES, CONDITIONS, BRAND_TIERS
from users import ITEM_CATALOG

# set this to your area; listings are scattered within ~15 miles of it
CENTER_LAT, CENTER_LON = 40.0, -75.0
PRICE_MEDIAN = {"furniture": 120, "kitchen": 35, "electronics": 90,
                "decor": 40, "bedding": 30}


def make_listing(rng: np.random.Generator, listing_id: str) -> Listing:
    cat = str(rng.choice(CATEGORIES, p=[0.35, 0.2, 0.2, 0.15, 0.1]))
    item = str(rng.choice(ITEM_CATALOG[cat]))
    cond_i = int(rng.choice(3, p=[0.25, 0.45, 0.30]))
    condition = CONDITIONS[cond_i]

    # seller: 90% "normal" sellers near 100%, 10% with a long negative tail
    neg = rng.exponential(0.8) if rng.random() < 0.9 else rng.exponential(5.0)
    pct = float(round(np.clip(100.0 - neg, 80.0, 100.0), 1))
    score = int(rng.lognormal(mean=4.0, sigma=1.8))
    business = bool(rng.random() < (0.6 if score > 1000 else 0.15))

    brand = str(rng.choice(BRAND_TIERS, p=[0.35, 0.25, 0.28, 0.12]))

    # damage mentions are more likely the worse the stated condition
    kw_p = {"damage": [0.7, 0.3, 0.05][cond_i]}
    keywords = [k for k, p in kw_p.items() if rng.random() < p]

    cond_mult = [0.55, 0.8, 1.0][cond_i] * [0.8, 0.9, 1.1, 1.6][BRAND_TIERS.index(brand)]
    price = float(round(PRICE_MEDIAN[cat] * cond_mult * rng.lognormal(0, 0.5), 2))

    r_miles, theta = 15 * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
    lat = CENTER_LAT + (r_miles * np.sin(theta)) / 69.0
    lon = CENTER_LON + (r_miles * np.cos(theta)) / (69.0 * np.cos(np.radians(CENTER_LAT)))

    title_bits = [brand.capitalize() if brand != "unbranded" else "", item]
    return Listing(
        listing_id=listing_id,
        title=" ".join(b for b in title_bits if b),
        category=cat, condition=condition,
        seller_feedback_pct=pct, seller_feedback_score=score,
        business_seller=business,
        returns_accepted=bool(rng.random() < (0.8 if business else 0.3)),
        brand_tier=brand, keywords=keywords,
        price=price, lat=float(round(lat, 5)), lon=float(round(lon, 5)),
    )


def make_listings(n: int, rng: np.random.Generator) -> list:
    return [make_listing(rng, f"L{i:05d}") for i in range(n)]