# Personalized Secondhand Marketplace Recommender

**CS 5100 Project** · Hannah Bang, Manavkrishna Manoj, Tarun Sethi

## 1. Overview

Buying secondhand saves money and reduces waste, but it is full of compromises. Listings are hard to compare, photos may not match the stated condition, and the best deal depends on the buyer: someone who only walks needs nearby pickups, while someone with a car on a tight budget may drive farther to save money. Buyers furnishing a home also need several items at once, so the best single listing for each item does not always add up to the best overall purchase within their budget.

This project recommends the best **combination** of listings for a buyer's whole shopping list. It runs in four steps:

```
Listings ──► 1. Condition from photos ──► 2. Hard constraints ──► 3. Matching model ──► 4. Combination search ──► Recommendations
              (CNN)                         (filtering)               (pairwise ranking)      (simulated annealing)
```

| Step | Approach | Status |
|---|---|---|
| 1. Condition from photos | CNN trained from scratch to predict condition from listing photos | Not started |
| 2. Hard constraints | Removes listings not on the shopping list, over budget, or too far to fit in the buyer's pickup trip | **Done** |
| 3. Matching model | Pairwise logistic regression (Bradley-Terry) trained on "which would you rather buy?" choices | Mock training data done |
| 4. Combination search | Simulated annealing picks one listing per item, trading off match score against price and trip distance with buyer-controlled sliders | **Next up** |

All AI methods are implemented from scratch in Python, NumPy, and PyTorch, with no pretrained models. See the **[project proposal](docs/proposal.md)** for the full methods, evaluation plan, and related work.

### What works so far
- **Onboarding app:** a web form where the buyer enters their budget, transport modes, home address, pickup-trip limit, shopping list, and how much condition, seller trust, and brand matter to them (1 to 5). Every question is required.
- **Hard constraints:** the buyer's address is converted to coordinates with the free [US Census geocoder](https://geocoding.geo.census.gov/geocoder/). A listing is kept only if it is on the shopping list, costs no more than the budget, and going there and back from home fits in the trip limit. The app then shows how many listings are left for each item.
- **Mock data:** simulated buyers, listings with realistic titles, and "which would you rather buy?" answers for training the matching model.

## 2. Setting Up the Environment

Requires **Python 3.12 or newer** (tested on 3.14) and an internet connection (the app looks up addresses online).

```
git clone https://github.com/hannahbang1108/Personalized-Secondhand-Marketplace-Recommender.git
cd Personalized-Secondhand-Marketplace-Recommender

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

## 3. Running the Code

**Generate the mock data:**

```
python generate_mock_data.py
```

This writes four files to `data/`. The same `--seed` always produces the same data, so results are reproducible. Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--buyers` | 400 | Number of simulated buyers |
| `--listings` | 3000 | Number of mock listings |
| `--questions-per-buyer` | 40 | "Which would you rather buy?" questions each buyer answers |
| `--consistency` | 1.5 | How reliably buyers pick the listing they truly prefer |
| `--cold-start-share` | 0.2 | Share of buyers held out entirely |
| `--val-share` | 0.1 | Share of answers used for validation |
| `--test-share` | 0.2 | Share of answers used for the final test |
| `--seed` | 0 | Random seed |

**Run the app:**

```
python -m streamlit run app.py
```

This opens the app in your browser. The mock listings are scattered within 15 miles of Philadelphia, so use an address in that area, for example `1400 John F Kennedy Blvd, Philadelphia, PA` with a budget of 300, a trip limit of 16 miles, and a shopping list of desk, sofa, and tv.

Model training, the combination search, and the evaluation scripts will be added here as they are built.

## 4. Code Organization

| File | What it does |
|---|---|
| `app.py` | Streamlit app: onboarding form and results. |
| `constraints.py` | Step 2: address lookup and the hard-constraint filter. |
| `schema.py` | Defines the `Listing` and `Buyer` formats and turns a listing into the feature vector the matching model uses. Every data source (mock now, real data later) must produce this format. |
| `mock_listings.py` | Mock listings with realistic titles, prices, seller details, and locations. To be replaced by real listings later. |
| `mock_buyers.py` | Simulated buyers: their onboarding answers, their true preferences, and their "which would you rather buy?" answers. |
| `generate_mock_data.py` | Generates buyers, listings, and answers, splits them for training and evaluation, and saves everything to `data/`. |
| `requirements.txt` | Python packages to install. |
| `docs/proposal.md` | Full project proposal. |

### Generated data (`data/`)

| File | Contents | Used by |
|---|---|---|
| `listings.json` | Listing records | Constraints, model, and search |
| `buyers.json` | Simulated onboarding answers: priorities, transport modes, budget, shopping list | Model and search |
| `comparisons.csv` | Every "which would you rather buy?" answer, with a `split` column: `train` (fit the model), `val` (tune settings), `test` (final accuracy, used once), `cold_start` (buyers held out entirely) | Model training and evaluation |
| `true_weights.json` | Each simulated buyer's true preferences (the answer key) | **Evaluation only**, never training |

### Listing features in the match score

Each buyer's true weight for a feature is a base weight scaled by their rating (1 to 5) for the priority that controls it:

| Priority | Features |
|---|---|
| Condition | `condition`, `mentions_damage` |
| Seller trust | `seller_negative_feedback`, `seller_review_count`, `business_seller`, `returns_accepted` |
| Brand | `brand_budget`, `brand_mid`, `brand_premium` (compared with unbranded) |

Price and distance are not part of the match score; they are handled by the hard constraints and the combination search.
