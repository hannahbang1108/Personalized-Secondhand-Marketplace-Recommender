"""
Shared schema for listings and buyers.

Every data source (mock generator now, eBay adapter later) must produce
Listing objects with these fields. Everything downstream (features, matching
model, search) reads only this schema, never source-specific fields.
"""
from dataclasses import dataclass, field
import numpy as np

# ---------- vocabularies ----------
CATEGORIES = ["furniture", "kitchen", "electronics", "decor", "bedding"]
CONDITIONS = ["fair", "good", "like_new"]            # ordinal 0, 1, 2
BRAND_TIERS = ["unbranded", "budget", "mid", "premium"]
KEYWORDS = ["damage"]    # description mentions scratches, wear, stains, etc.

TRANSPORT_MODES = ["walk", "transit", "car"]
PRIORITY_KEYS = ["condition", "seller_trust", "brand"]


@dataclass
class Listing:
    listing_id: str
    title: str
    category: str
    condition: str                 # seller-stated for now; CNN prediction later
    seller_feedback_pct: float     # eBay: seller.feedbackPercentage
    seller_feedback_score: int     # eBay: seller.feedbackScore
    business_seller: bool          # eBay: seller.sellerAccountType == BUSINESS
    returns_accepted: bool         # eBay: return terms
    brand_tier: str
    keywords: list = field(default_factory=list)
    # Used by hard constraints and search only. NOT part of the match score.
    price: float = 0.0
    lat: float = 0.0
    lon: float = 0.0


@dataclass
class Buyer:
    """Only what the app's onboarding form asks a real user."""
    buyer_id: str
    name: str
    priorities: dict               # PRIORITY_KEYS -> int 1..5
    transport: list                # modes the buyer is willing to use, e.g. ["walk", "transit"]
    budget: float
    shopping_list: list            # [{"item": "desk", "category": "furniture"}]


# ---------- feature encoders (the only thing the model sees) ----------
# Category is not a feature: buyers only compare listings within one category.
# Brand is one-hot with "unbranded" as the baseline (all zeros).
LISTING_FEATURE_NAMES = (
    ["condition", "seller_negative", "seller_score_log", "business_seller",
     "returns_accepted"]
    + [f"brand_{b}" for b in BRAND_TIERS if b != "unbranded"]
    + [f"kw_{k}" for k in KEYWORDS]
)

# Price, distance, transport, and budget are handled by constraints and search.
BUYER_FEATURE_NAMES = (
    [f"prio_{k}" for k in PRIORITY_KEYS]
)


def listing_features(l: Listing) -> np.ndarray:
    """Encode a listing as a fixed-length vector. Rough scaling only;
    standardize with training-set statistics before fitting a model."""
    f = [
        CONDITIONS.index(l.condition) / 2.0,
        np.log1p(100.0 - l.seller_feedback_pct),      # 0 at 100% positive
        np.log1p(l.seller_feedback_score) / 10.0,
        float(l.business_seller),
        float(l.returns_accepted),
    ]
    f += [float(l.brand_tier == b) for b in BRAND_TIERS if b != "unbranded"]
    f += [float(k in l.keywords) for k in KEYWORDS]
    return np.array(f, dtype=np.float64)


def buyer_features(b: Buyer) -> np.ndarray:
    f = [b.priorities[k] / 5.0 for k in PRIORITY_KEYS]
    return np.array(f, dtype=np.float64)