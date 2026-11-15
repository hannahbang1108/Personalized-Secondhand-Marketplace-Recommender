"""
Generate the mock dataset for the matching model.

Run it with:
    python generate_mock_data.py
    python generate_mock_data.py --buyers 500 --consistency 3

It writes four files into data/:
    listings.json       the listings
    buyers.json         each buyer's onboarding-form answers
    comparisons.csv     every rating a buyer gave a listing, from 1 to 5
    true_weights.json   each buyer's true preferences (the answer key, never
                        used for training)

Each row of comparisons.csv has a `split` column, following the course notes
on model training:
    train       used to fit the model's weights
    val         used to tune settings like the learning rate
    test        used once at the end, for a fair final accuracy number
    cold_start  answers from buyers held out entirely, to test how well the
                model handles brand-new buyers
"""
import argparse
import csv
import json
import os
from dataclasses import asdict
import numpy as np

from mock_buyers import make_buyer, make_true_weights, simulate_choices
from mock_listings import make_listings


def main():
    settings = read_settings()
    rng = np.random.default_rng(settings.seed)

    listings = make_listings(settings.listings, rng)
    home_rng = rng.spawn(1)[0]
    buyers = [make_buyer(rng, f"U{i:04d}", home_rng) for i in range(settings.buyers)]
    true_weights = {buyer.buyer_id: make_true_weights(buyer) for buyer in buyers}
    choices = simulate_choices(buyers, true_weights, listings, rng,
                               questions_per_buyer=settings.questions_per_buyer,
                               consistency=settings.consistency)

    assign_splits(choices, buyers, settings, rng)
    save_files(settings.output_folder, listings, buyers, choices, true_weights)
    print_summary(listings, buyers, choices)


def read_settings():
    parser = argparse.ArgumentParser(description="Generate mock data for the matching model.")
    parser.add_argument("--buyers", type=int, default=400)
    parser.add_argument("--listings", type=int, default=3000)
    parser.add_argument("--questions-per-buyer", type=int, default=40)
    parser.add_argument("--consistency", type=float, default=1.5,
                        help="how closely buyers' ratings follow their true preferences")
    parser.add_argument("--cold-start-share", type=float, default=0.2,
                        help="share of buyers held out entirely")
    parser.add_argument("--val-share", type=float, default=0.1)
    parser.add_argument("--test-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0,
                        help="same seed = same data every time")
    parser.add_argument("--output-folder", default="data")
    return parser.parse_args()


def assign_splits(choices, buyers, settings, rng):
    """Label every choice as train, val, test, or cold_start."""
    buyer_ids = [buyer.buyer_id for buyer in buyers]
    num_held_out = int(len(buyer_ids) * settings.cold_start_share)
    held_out_buyers = set(rng.choice(buyer_ids, size=num_held_out, replace=False))

    for choice in choices:
        if choice["buyer_id"] in held_out_buyers:
            choice["split"] = "cold_start"
            continue
        roll = rng.random()
        if roll < settings.val_share:
            choice["split"] = "val"
        elif roll < settings.val_share + settings.test_share:
            choice["split"] = "test"
        else:
            choice["split"] = "train"


def save_files(folder, listings, buyers, choices, true_weights):
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, "listings.json"), "w") as f:
        json.dump([asdict(listing) for listing in listings], f, indent=1)

    with open(os.path.join(folder, "buyers.json"), "w") as f:
        json.dump([asdict(buyer) for buyer in buyers], f, indent=1)

    with open(os.path.join(folder, "true_weights.json"), "w") as f:
        json.dump(true_weights, f, indent=1)

    with open(os.path.join(folder, "comparisons.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(choices[0]))
        writer.writeheader()
        writer.writerows(choices)


def print_summary(listings, buyers, choices):
    """Print a few quick checks that the data looks reasonable."""
    split_sizes = {}
    for choice in choices:
        split_sizes[choice["split"]] = split_sizes.get(choice["split"], 0) + 1

    stars = np.array([c["rating"] for c in choices])
    true_scores = np.array([c["true_score"] for c in choices])
    star_shares = {s: f"{np.mean(stars == s):.0%}" for s in range(1, 6)}

    # How closely the ratings follow the buyers' true preferences. Because
    # buyers aren't perfectly consistent, this is below 1, and it's the
    # target for the model.
    agreement = np.corrcoef(stars, true_scores)[0, 1]

    print(f"{len(listings)} listings, {len(buyers)} buyers, {len(choices)} ratings")
    print(f"split sizes: {split_sizes}")
    print(f"share of each star rating (1 to 5): {star_shares}")
    print(f"correlation between ratings and true scores: {agreement:.2f}")
    print(f"\nexample buyer: {json.dumps(asdict(buyers[0]))}")


if __name__ == "__main__":
    main()
