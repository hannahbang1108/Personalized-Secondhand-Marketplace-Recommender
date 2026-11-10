# Personalized Secondhand Marketplace Recommender

**CS 5100 Project** · Hannah Bang, Manavkrishna Manoj, Tarun Sethi

A recommender that matches secondhand listings to each buyer's preferences, checks item condition from listing photos, and finds the best combination of listings for a whole shopping list, with sliders that control how much price and distance matter.

For the motivation, methods, evaluation plan, and related work, see the **[project proposal](docs/proposal.md)**.

## Pipeline

```
Listings ──► 1. Condition from photos ──► 2. Hard constraints ──► 3. Matching model ──► 4. Combination search ──► Recommendations
              (CNN)                         (filtering)               (Gaussian process)      (simulated annealing)
```

| Step | Approach |
|---|---|
| 1. Condition from photos | CNN trained from scratch to predict condition from listing photos |
| 2. Hard constraints | Removes listings over the buyer's budget or beyond their transport range |
| 3. Matching model | Gaussian process regression over buyer-listing pairs, giving a match score and its uncertainty |
| 4. Combination search | Simulated annealing to pick one listing per shopping-list item |

### Matching model

The Gaussian process predicts how well a listing matches a buyer from how similar that buyer-listing pair is to pairs that have already been rated. Similarity is a product kernel: buyer similarity times listing similarity, each an RBF kernel. Every prediction comes with a mean (the match score) and a variance (how sure the model is), and the posterior is exactly Gaussian, so no approximation is needed.

An exact Gaussian process slows down as the number of ratings grows, since its cost grows with the cube of the training set size. To scale, the model uses a sparse Gaussian process with inducing points: a few hundred representative points, chosen by k-means, stand in for the full training set.

If time allows, we will also train a neural network matching model and compare it against the Gaussian process.

All AI methods are implemented from scratch in Python, NumPy, and PyTorch, with no pretrained models.

**Current status:** mock data generation for the matching model.

## Getting Started

Requires Python 3.9+.

```
pip install -r requirements.txt
python generate_mock_data.py
```

This creates a `data/` folder with mock buyers, listings, and ratings. Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--buyers` | 400 | Number of simulated buyers |
| `--listings` | 3000 | Number of mock listings |
| `--ratings-per-buyer` | 40 | Listings each buyer rates |
| `--rating-noise` | 0.5 | How much ratings vary from a buyer's true preferences |
| `--cold-start-share` | 0.2 | Share of buyers held out entirely |
| `--val-share` | 0.1 | Share of ratings used for validation |
| `--test-share` | 0.2 | Share of ratings used for the final test |
| `--seed` | 0 | Random seed; the same seed gives the same data |

## Repository Structure

| File | What it does |
|---|---|
| `schema.py` | Defines the `Listing` and `Buyer` formats and turns a listing into the feature vector the model uses. Every data source (mock now, real-data-based later) must produce this format. |
| `mock_buyers.py` | Creates simulated buyers from onboarding-form answers, sets each buyer's true preferences from their priority ratings, and simulates their ratings. |
| `mock_listings.py` | Temporary fake listings. To be replaced by listings sampled from real datasets. |
| `generate_mock_data.py` | The script you run. Generates buyers, listings, and ratings, splits them for training and evaluation, and saves everything to `data/`. |
| `docs/proposal.md` | Full project proposal: motivation, approach, methods, evaluation plan, related work. |

### Generated data (`data/`)

| File | Contents | Used by |
|---|---|---|
| `listings.json` | Listing records | Model and search |
| `buyers.json` | Onboarding-form answers: priorities, transport modes, budget, shopping list | Model, constraints, and search |
| `ratings.csv` | Every buyer's rating of a listing, with a `split` column: `train` (fit the model), `val` (tune kernel settings), `test` (final accuracy, used once), `cold_start` (buyers held out entirely) | Model training and evaluation |
| `true_weights.json` | Each buyer's true preferences (the answer key) | **Evaluation only**, never training |

### Listing features in the match score

Each buyer's true weight for a feature is a base weight scaled by their rating (1 to 5) for the priority that controls it:

| Priority | Features |
|---|---|
| Condition | `condition`, `mentions_damage` |
| Seller trust | `seller_negative_feedback`, `seller_review_count`, `business_seller`, `returns_accepted` |
| Brand | `brand_budget`, `brand_mid`, `brand_premium` (compared with unbranded) |

Price and distance are not part of the match score; they are handled by the hard constraints and the combination search.
