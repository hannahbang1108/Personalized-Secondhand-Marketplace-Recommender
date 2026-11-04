"""
Generate the mock dataset for the matching model.

    python generate_mock_data.py                      # procedural personas
    python generate_mock_data.py --users 500 --beta 2.0

Outputs (in data/):
    listings.json       listing records (schema.Listing)
    buyers.json         onboarding-form answers the model may see (schema.Buyer)
    ground_truth.json   each buyer's true weights (answer key), for evaluation only
    comparisons.csv     pairwise choices with a `split` column:
                          train       fit the model's weights
                          val         tune hyperparameters (learning rate, etc.)
                          test        unbiased final accuracy, used once at the end
                          cold_start  all pairs from buyers held out entirely
                        train/val/test come from the same buyers; cold_start
                        buyers never appear in training.
"""
import argparse, csv, json, os
from dataclasses import asdict
import numpy as np

from users import make_buyer, make_ground_truth, simulate_comparisons, true_utility
from mock_listings_stub import make_listings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=400)
    ap.add_argument("--listings", type=int, default=3000)
    ap.add_argument("--pairs-per-user", type=int, default=40)
    ap.add_argument("--beta", type=float, default=1.5, help="choice consistency")
    ap.add_argument("--cold-start-share", type=float, default=0.2)
    ap.add_argument("--val-share", type=float, default=0.1)
    ap.add_argument("--test-share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    listings = make_listings(args.listings, rng)
    # To use an LLM, pass call_llm=your_function (prompt -> text) to make_buyer.
    buyers = [make_buyer(rng, f"U{i:04d}") for i in range(args.users)]
    truths = {b.buyer_id: make_ground_truth(b) for b in buyers}
    rows = simulate_comparisons(buyers, truths, listings, rng,
                                n_per_user=args.pairs_per_user, beta=args.beta)

    # splits (see "Model Training" in the course notes): train / validation /
    # test from the same buyers, plus whole buyers held out for cold start
    ids = [b.buyer_id for b in buyers]
    cold = set(rng.choice(ids, size=int(len(ids) * args.cold_start_share), replace=False))
    for r in rows:
        if r["buyer_id"] in cold:
            r["split"] = "cold_start"
            continue
        u = rng.random()
        r["split"] = ("val" if u < args.val_share else
                      "test" if u < args.val_share + args.test_share else "train")

    with open(f"{args.out}/listings.json", "w") as f:
        json.dump([asdict(l) for l in listings], f, indent=1)
    with open(f"{args.out}/buyers.json", "w") as f:
        json.dump([asdict(b) for b in buyers], f, indent=1)
    with open(f"{args.out}/ground_truth.json", "w") as f:
        json.dump(truths, f, indent=1)
    with open(f"{args.out}/comparisons.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- sanity report ----
    by_split = {}
    for r in rows:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    p = np.array([r["true_p_a"] for r in rows])
    ceiling = np.mean(np.maximum(p, 1 - p))  # best possible accuracy
    print(f"{len(listings)} listings, {len(buyers)} buyers, {len(rows)} comparisons")
    print("split sizes:", by_split)
    print(f"share of A chosen: {np.mean([r['chose_a'] for r in rows]):.3f} (should be ~0.5)")
    print(f"accuracy ceiling (always pick the truly better listing): {ceiling:.3f}")
    print(f"near coin-flip pairs (0.4 < p < 0.6): {np.mean((p > 0.4) & (p < 0.6)):.3f}")
    b = buyers[0]
    print("\nexample buyer (visible):", json.dumps(asdict(b)))


if __name__ == "__main__":
    main()