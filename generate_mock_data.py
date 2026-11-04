"""
Generate the mock dataset for the matching model.

Run it with:
    python generate_mock_data.py
    python generate_mock_data.py --buyers 500 --consistency 3

It writes four files into data/:
    listings.json       the listings
    buyers.json         each buyer's onboarding-form answers
    comparisons.csv     every "which would you rather buy?" answer
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
    # To use an LLM, pass ask_llm=your_function to make_buyer (see mock_buyers.py).
    buyers = [make_buyer(rng, f"U{i:04d}") for i in range(settings.buyers)]
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
                        help="how reliably buyers pick the listing they truly prefer")
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

    prob_chose_a = np.array([c["true_prob_chose_a"] for c in choices])
    share_chose_a = np.mean([c["chose_a"] for c in choices])

    # The best any model could possibly do is always predict the listing the
    # buyer is more likely to choose. Because buyers sometimes pick the
    # "wrong" one, this is below 100%, and it's the target for the model.
    best_possible_accuracy = np.mean(np.maximum(prob_chose_a, 1 - prob_chose_a))

    print(f"{len(listings)} listings, {len(buyers)} buyers, {len(choices)} comparisons")
    print(f"split sizes: {split_sizes}")
    print(f"listing A was chosen {share_chose_a:.1%} of the time (should be near 50%)")
    print(f"best possible model accuracy: {best_possible_accuracy:.1%}")
    print(f"\nexample buyer: {json.dumps(asdict(buyers[0]))}")


if __name__ == "__main__":
    main()