"""
Simulated buyers for training and evaluating the matching model.

Each buyer is just their onboarding-form answers (schema.Buyer). Their true
preferences are set in advance from those answers by fixed rules, then used
to simulate "which would you rather buy?" choices.

Pipeline:
  1. make_buyer()           fill out the form: budget, transport modes, shopping
                            list, priorities (procedurally or with an LLM)
  2. make_ground_truth()    turn the form answers into the buyer's true weights
  3. simulate_comparisons() simulate pairwise choices from those true weights

The matching model trains on buyer_features(buyer) and the comparisons.
ground_truth is the answer key, used for evaluation only.
"""
import json
from typing import Callable, Optional
import numpy as np

from schema import (Buyer, Listing, TRANSPORT_MODES, PRIORITY_KEYS,
                    LISTING_FEATURE_NAMES, listing_features)

ITEM_CATALOG = {
    "furniture": ["desk", "office chair", "sofa", "bed frame", "dresser",
                  "bookshelf", "dining table", "coffee table"],
    "kitchen": ["microwave", "blender", "cookware set", "coffee maker", "toaster oven"],
    "electronics": ["monitor", "tv", "speaker", "printer"],
    "decor": ["floor lamp", "rug", "mirror", "wall art"],
    "bedding": ["duvet", "mattress topper", "curtains"],
}

FIRST_NAMES = ["Maya", "Jordan", "Priya", "Luis", "Chen", "Amara", "Sam", "Elena",
               "Omar", "Grace", "Diego", "Hana", "Noah", "Fatima", "Theo", "Ines",
               "Kwame", "Lena", "Ravi", "Zoe"]

# ---------------------------------------------------------------------------
# 1. Buyers (form answers)
# ---------------------------------------------------------------------------
def sample_form_basics(rng: np.random.Generator, buyer_id: str) -> dict:
    n_items = int(rng.integers(2, 6))
    all_items = [(item, cat) for cat, items in ITEM_CATALOG.items() for item in items]
    cat_p = {"furniture": 0.40, "kitchen": 0.20, "electronics": 0.15, "decor": 0.15, "bedding": 0.10}
    p = np.array([cat_p[c] / len(ITEM_CATALOG[c]) for _, c in all_items])
    picks = rng.choice(len(all_items), size=n_items, replace=False, p=p / p.sum())
    # each mode the buyer is willing to use; at least one
    willing = {"walk": 0.7, "transit": 0.5, "car": 0.45}
    transport = [m for m in TRANSPORT_MODES if rng.random() < willing[m]]
    if not transport:
        transport = [str(rng.choice(TRANSPORT_MODES))]
    return {
        "buyer_id": buyer_id,
        "transport": transport,
        "budget": float(round(rng.uniform(150, 2000), -1)),
        "shopping_list": [{"item": all_items[i][0], "category": all_items[i][1]}
                          for i in picks],
    }


def fill_procedural(rng: np.random.Generator) -> dict:
    return {
        "name": str(rng.choice(FIRST_NAMES)),
        "priorities": {k: int(rng.integers(1, 6)) for k in PRIORITY_KEYS},
    }


PERSONA_PROMPT = """You are filling out an onboarding form as one realistic secondhand-marketplace buyer, for a research simulation.

These answers are already fixed:
{facts}

Return ONLY a JSON object, no prose and no code fences, with exactly these keys:
- "name": a first name
- "priorities": an object with integer ratings 1-5 for each of {priority_keys}, where 5 means "matters a lot"

Make the answers believable for someone with this budget and shopping list, and avoid rating everything 3 or 4."""


def validate_llm_fields(d: dict) -> dict:
    """Raise ValueError if the LLM output doesn't match the schema."""
    if set(d) != {"name", "priorities"}:
        raise ValueError(f"wrong keys: {sorted(d)}")
    pr = d["priorities"]
    if set(pr) != set(PRIORITY_KEYS) or not all(
            isinstance(v, int) and 1 <= v <= 5 for v in pr.values()):
        raise ValueError(f"bad priorities: {pr}")
    if not isinstance(d["name"], str):
        raise ValueError("name must be a string")
    return d


def fill_with_llm(basics: dict, call_llm: Callable[[str], str],
                  rng: np.random.Generator, max_retries: int = 3) -> dict:
    facts = {k: v for k, v in basics.items() if k != "buyer_id"}
    prompt = PERSONA_PROMPT.format(facts=json.dumps(facts, indent=2),
                                   priority_keys=PRIORITY_KEYS)
    for _ in range(max_retries):
        try:
            text = call_llm(prompt).strip().removeprefix("```json").removesuffix("```")
            return validate_llm_fields(json.loads(text))
        except (ValueError, json.JSONDecodeError):
            continue
    return fill_procedural(rng)  # fall back rather than crash


def make_buyer(rng: np.random.Generator, buyer_id: str,
               call_llm: Optional[Callable[[str], str]] = None) -> Buyer:
    basics = sample_form_basics(rng, buyer_id)
    rest = fill_with_llm(basics, call_llm, rng) if call_llm else fill_procedural(rng)
    return Buyer(**basics, **rest)


# ---------------------------------------------------------------------------
# 2. True preferences, set in advance from the form answers
# ---------------------------------------------------------------------------
def make_ground_truth(b: Buyer) -> dict:
    """True weight on each listing feature for this buyer."""
    p = {k: v / 3.0 for k, v in b.priorities.items()}   # 1..5 -> ~0.33..1.67
    w = dict.fromkeys(LISTING_FEATURE_NAMES, 0.0)

    # condition priority
    w["condition"] = 1.0 * p["condition"]
    w["kw_damage"] = -0.5 * p["condition"]
    # seller-trust priority
    w["seller_negative"] = -0.6 * p["seller_trust"]
    w["seller_score_log"] = 0.8 * p["seller_trust"]
    w["business_seller"] = 0.2 * p["seller_trust"]
    w["returns_accepted"] = 0.2 * p["seller_trust"]
    # brand priority (each tier compared with unbranded)
    w["brand_budget"], w["brand_mid"], w["brand_premium"] = (
        0.1 * p["brand"], 0.4 * p["brand"], 0.8 * p["brand"])
    return {"weights": w}


def true_utility(b: Buyer, gt: dict, l: Listing) -> float:
    x = listing_features(l)
    return float(np.dot([gt["weights"][n] for n in LISTING_FEATURE_NAMES], x))


# ---------------------------------------------------------------------------
# 3. Simulated pairwise choices
# ---------------------------------------------------------------------------
def simulate_comparisons(buyers, ground_truths, listings, rng, n_per_user=40,
                         beta=1.5, shopping_list_share=0.8):
    """Each buyer compares pairs of same-category listings.
    P(choose A) = sigmoid(beta * (u_A - u_B)). Higher beta = more consistent buyers."""
    by_cat = {}
    for l in listings:
        by_cat.setdefault(l.category, []).append(l)

    rows = []
    for b in buyers:
        gt = ground_truths[b.buyer_id]
        own_cats = [x["category"] for x in b.shopping_list]
        for _ in range(n_per_user):
            cat = (rng.choice(own_cats) if rng.random() < shopping_list_share
                   else rng.choice(list(by_cat)))
            i, j = rng.choice(len(by_cat[cat]), size=2, replace=False)
            a, c = by_cat[cat][i], by_cat[cat][j]
            diff = true_utility(b, gt, a) - true_utility(b, gt, c)
            p_a = 1.0 / (1.0 + np.exp(-beta * diff))
            rows.append({"buyer_id": b.buyer_id, "listing_a": a.listing_id,
                         "listing_b": c.listing_id, "chose_a": int(rng.random() < p_a),
                         "true_p_a": round(float(p_a), 4)})   # true_p_a: eval only
    return rows