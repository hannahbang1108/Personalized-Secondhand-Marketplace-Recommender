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


# Brands for each brand tier, by category.
BRANDS = {
    "furniture": {"budget": ["IKEA", "Mainstays", "Sauder"],
                  "mid": ["Wayfair", "Ashley", "Threshold"],
                  "premium": ["West Elm", "Pottery Barn", "Crate & Barrel"]},
    "kitchen": {"budget": ["Hamilton Beach", "Black+Decker", "Mainstays"],
                "mid": ["Cuisinart", "Ninja", "Oster"],
                "premium": ["KitchenAid", "Breville"]},
    "electronics": {"budget": ["Insignia", "Onn"],
                    "mid": ["Samsung", "LG", "HP"],
                    "premium": ["Sony", "Bose", "Apple"]},
    "decor": {"budget": ["Mainstays", "IKEA"],
              "mid": ["Threshold", "Wayfair"],
              "premium": ["West Elm", "CB2", "Pottery Barn"]},
    "bedding": {"budget": ["Mainstays", "Amazon Basics"],
                "mid": ["Threshold", "Linenspa"],
                "premium": ["Brooklinen", "Parachute", "Tempur-Pedic"]},
}

# A few ways sellers describe each item.
ITEM_NAMES = {
    "desk": ["Computer Desk", "Writing Desk with Drawer", 'Standing Desk 48"'],
    "office chair": ["Ergonomic Office Chair", "Mesh Desk Chair", "Swivel Office Chair"],
    "sofa": ["3-Seat Sofa", "Loveseat", "Sectional Sofa"],
    "bed frame": ["Queen Bed Frame", "Full Platform Bed Frame", "Twin Metal Bed Frame"],
    "dresser": ["6-Drawer Dresser", "4-Drawer Dresser", "Wood Dresser"],
    "bookshelf": ["5-Shelf Bookcase", "Ladder Bookshelf", "Cube Bookshelf"],
    "dining table": ["Dining Table for 4", "Extendable Dining Table", "Round Dining Table"],
    "coffee table": ["Coffee Table", "Lift-Top Coffee Table", "Glass Coffee Table"],
    "microwave": ["Countertop Microwave 1.1 cu ft", "900W Microwave", "Compact Microwave"],
    "blender": ["Countertop Blender", "Personal Blender", "64oz Blender"],
    "cookware set": ["10-Piece Cookware Set", "Nonstick Pots and Pans", "Stainless Cookware Set"],
    "coffee maker": ["12-Cup Coffee Maker", "Drip Coffee Maker", "Single Serve Coffee Maker"],
    "toaster oven": ["Toaster Oven", "Convection Toaster Oven", "4-Slice Toaster Oven"],
    "monitor": ['24" 1080p Monitor', '27" Monitor', '27" 4K Monitor'],
    "tv": ['32" TV', '43" Smart TV', '55" 4K Smart TV'],
    "speaker": ["Bluetooth Speaker", "Bookshelf Speakers (Pair)", "Soundbar"],
    "printer": ["Wireless Printer", "All-in-One Inkjet Printer", "Laser Printer"],
    "floor lamp": ["Floor Lamp", "Arc Floor Lamp", "LED Floor Lamp"],
    "rug": ["5x7 Area Rug", "8x10 Area Rug", "Runner Rug"],
    "mirror": ["Full Length Mirror", "Round Wall Mirror", "Floor Mirror"],
    "wall art": ["Framed Wall Art", "Canvas Print", "Set of 3 Framed Prints"],
    "duvet": ["Queen Duvet Cover Set", "Down Alternative Duvet", "Full/Queen Comforter"],
    "mattress topper": ["Queen Memory Foam Topper", '3" Mattress Topper', "Twin XL Mattress Topper"],
    "curtains": ['Blackout Curtains 84"', "Linen Curtains (2 Panels)", "Sheer Curtains"],
}

# Optional endings, depending on condition.
TITLE_ENDINGS = {
    "like_new": ["Like New", "Barely Used", "Moving Sale", "Local Pickup"],
    "good": ["Good Condition", "Moving Sale", "Local Pickup", "Must Go"],
    "fair": ["Some Wear", "As Is", "Must Go", "Local Pickup"],
}


def make_listings(count, rng):
    # Titles use their own random stream so they don't change the rest of the
    # generated data (buyers, comparisons) for the same seed.
    title_rng = rng.spawn(1)[0]
    return [make_listing(f"L{i:05d}", rng, title_rng) for i in range(count)]


def make_listing(listing_id, rng, title_rng):
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
    return Listing(
        listing_id=listing_id,
        title=make_title(item, category, brand, condition, title_rng),
        item=item,
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


def make_title(item, category, brand_tier, condition, rng):
    """e.g. "IKEA 6-Drawer Dresser - Some Wear" """
    title = str(rng.choice(ITEM_NAMES[item]))
    if brand_tier != "unbranded":
        title = f"{rng.choice(BRANDS[category][brand_tier])} {title}"
    if rng.random() < 0.5:
        title += f" - {rng.choice(TITLE_ENDINGS[condition])}"
    return title


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