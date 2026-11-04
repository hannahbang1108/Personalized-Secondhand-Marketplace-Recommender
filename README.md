# Personalized Secondhand Marketplace Recommender

**CS 5100 Project** · Hannah Bang, Manavkrishna Manoj, Tarun Sethi

A recommender that matches secondhand listings to each buyer's preferences, checks item condition from listing photos, and finds the best combination of listings for a whole shopping list, with sliders that control how much price and distance matter.

For the motivation, methods, evaluation plan, and related work, see the **[project proposal](docs/proposal.md)**.

## Pipeline

```
Listings ──► 1. Condition from photos ──► 2. Hard constraints ──► 3. Matching model ──► 4. Combination search ──► Recommendations
              (CNN)                         (filtering)               (pairwise ranking)      (simulated annealing)
```

All AI methods are implemented from scratch in Python, NumPy, and PyTorch, with no pretrained models.

**Current status:** mock data generation for the matching model.

## Getting Started

Requires Python 3.9+.

```
pip install -r requirements.txt
python generate_mock_data.py
```

This creates a `data/` folder with mock buyers, listings, and pairwise choices. Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--users` | 400 | Number of simulated buyers |
| `--listings` | 3000 | Number of mock listings |
| `--pairs-per-user` | 40 | Comparisons each buyer answers |
| `--beta` | 1.5 | How consistent buyers are (higher = fewer "wrong" picks) |
| `--val-share` | 0.1 | Share of pairs used for validation |
| `--test-share` | 0.2 | Share of pairs used for the final test |
| `--seed` | 0 | Random seed, for reproducible data |

## Repository Structure

| File | What it does |
|---|---|
| `schema.py` | Defines the `Listing` and `Buyer` formats and converts a listing into the feature vector the model uses. Every data source (mock now, eBay later) must produce this format. |
| `users.py` | Creates simulated buyers from onboarding-form answers, sets each buyer's true preference weights from their priority ratings, and simulates "which would you rather buy?" choices. Includes an optional LLM hook for filling in form answers. |
| `mock_listings_stub.py` | Temporary generator for fake listings with eBay-like distributions. To be replaced by Mercari-grounded listings and the eBay adapter. |
| `docs/proposal.md` | Full project proposal: motivation, approach, methods, evaluation plan, related work. |
| `generate_mock_data.py` | Main script. Generates buyers, listings, and comparisons, splits them for training and evaluation, and saves everything to `data/`. |

### Generated data (`data/`)

| File | Contents | Used by |
|---|---|---|
| `listings.json` | Listing records | Model and search |
| `buyers.json` | Onboarding-form answers: priorities, transport modes, budget, shopping list | Model, constraints, and search |
| `comparisons.csv` | Pairwise choices, with a `split` column: `train` (fit weights), `val` (tune hyperparameters), `test` (final accuracy, used once), `cold_start` (buyers held out entirely) | Model training and evaluation |
| `ground_truth.json` | Each buyer's true preference weights (the answer key) | **Evaluation only**, never training |

### Listing features in the match score

Each true weight scales with one of the buyer's three priority ratings (1 to 5):

| Priority | Features |
|---|---|
| Condition | `condition`, `kw_damage` |
| Seller trust | `seller_negative`, `seller_score_log`, `business_seller`, `returns_accepted` |
| Brand | `brand_budget`, `brand_mid`, `brand_premium` (compared with unbranded) |

Price and distance are excluded from the match score; they are handled by the hard constraints and the combination search.