"""
The shared data formats for the whole project.

Every data source (the mock generators now, the eBay adapter later) must
produce Listing and Buyer objects in exactly this format. The rest of the
system (matching model, constraints, search) only ever reads these objects,
so switching data sources never requires changes anywhere else.
"""
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------

# Everything a buyer can shop for, grouped by category.
ITEMS_BY_CATEGORY = {
    "furniture": ["desk", "office chair", "sofa", "bed frame", "dresser",
                  "bookshelf", "dining table", "coffee table"],
    "kitchen": ["microwave", "blender", "cookware set", "coffee maker", "toaster oven"],
    "electronics": ["monitor", "tv", "speaker", "printer"],
    "decor": ["floor lamp", "rug", "mirror", "wall art"],
    "bedding": ["duvet", "mattress topper", "curtains"],
}
CATEGORIES = list(ITEMS_BY_CATEGORY)

# Ordered from worst to best, so the position in the list is a condition score.
CONDITIONS = ["fair", "good", "like_new"]

BRAND_TIERS = ["unbranded", "budget", "mid", "premium"]

TRANSPORT_MODES = ["walk", "transit", "car"]

# The three things buyers rate from 1 (doesn't matter) to 5 (matters a lot).
PRIORITIES = ["condition", "seller_trust", "brand"]


# ---------------------------------------------------------------------------
# Data formats
# ---------------------------------------------------------------------------

@dataclass
class Listing:
    listing_id: str
    title: str
    category: str                  # one of CATEGORIES
    condition: str                 # one of CONDITIONS, as stated by the seller for now
    seller_feedback_percent: float # eBay: seller.feedbackPercentage, e.g. 99.2
    seller_review_count: int       # eBay: seller.feedbackScore
    business_seller: bool          # eBay: seller.sellerAccountType is BUSINESS
    returns_accepted: bool
    brand_tier: str                # one of BRAND_TIERS
    mentions_damage: bool          # description mentions scratches, wear, stains, etc.
    # The fields below are used by the hard constraints and the search.
    # They are deliberately NOT part of the match score.
    price: float
    latitude: float
    longitude: float


@dataclass
class Buyer:
    """A buyer's answers to the app's onboarding form."""
    buyer_id: str
    name: str
    priorities: dict               # each of PRIORITIES -> rating from 1 to 5
    transport: list                # modes they're willing to use, e.g. ["walk", "transit"]
    budget: float
    shopping_list: list            # e.g. [{"item": "desk", "category": "furniture"}]


# ---------------------------------------------------------------------------
# Feature extraction: turning a listing into numbers the model can use
# ---------------------------------------------------------------------------

# Names for each position in the feature vector, in order.
# Category isn't included because buyers only ever compare listings from the
# same category. Brand uses "unbranded" as the baseline: an unbranded listing
# has all three brand features set to 0.
LISTING_FEATURE_NAMES = [
    "condition",
    "seller_negative_feedback",
    "seller_review_count",
    "business_seller",
    "returns_accepted",
    "mentions_damage",
    "brand_budget",
    "brand_mid",
    "brand_premium",
]


def listing_to_features(listing: Listing) -> np.ndarray:
    """Convert a listing into a feature vector, in LISTING_FEATURE_NAMES order.

    Each feature is scaled to a similar range (roughly 0 to 1) so that no
    single feature dominates just because its raw numbers are bigger.
    """
    # fair = 0, good = 0.5, like new = 1
    condition_score = CONDITIONS.index(listing.condition) / (len(CONDITIONS) - 1)

    # Almost every eBay seller sits between 98% and 100% positive, so the raw
    # percentage barely varies. Measuring the distance below 100% on a log
    # scale spreads those sellers out: 100% -> 0, 99% -> 0.7, 90% -> 2.4.
    negative_feedback = np.log1p(100.0 - listing.seller_feedback_percent)

    # Review counts range from a handful to tens of thousands, so we use a log
    # scale (going from 10 to 100 reviews matters more than 10,010 to 10,100),
    # then divide by 10 to bring it into a range similar to the other features.
    review_count = np.log1p(listing.seller_review_count) / 10.0

    features = [
        condition_score,
        negative_feedback,
        review_count,
        float(listing.business_seller),
        float(listing.returns_accepted),
        float(listing.mentions_damage),
        float(listing.brand_tier == "budget"),
        float(listing.brand_tier == "mid"),
        float(listing.brand_tier == "premium"),
    ]
    return np.array(features)


def buyer_to_features(buyer: Buyer) -> np.ndarray:
    """Convert a buyer's priority ratings into numbers between 0.2 and 1."""
    return np.array([buyer.priorities[p] / 5.0 for p in PRIORITIES])