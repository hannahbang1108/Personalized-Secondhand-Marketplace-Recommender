# Personalized Secondhand Marketplace Recommender

**CS 5100 Project** · Hannah Bang, Manavkrishna Manoj, Tarun Sethi

## 1. Overview

Buying secondhand saves money and reduces waste, but it is full of compromises. Listings are hard to compare, photos may not match the stated condition, and the best deal depends on the buyer: someone who only walks needs nearby pickups, while someone with a car on a tight budget may drive farther to save money. Buyers furnishing a home also need several items at once, so the best single listing for each item does not always add up to the best overall purchase within their budget.

This project recommends the best **combination** of listings for a buyer's whole shopping list. It runs in four steps:

```
Listings ──► 1. Condition from photos ──► 2. Hard constraints ──► 3. Matching model ──► 4. Combination search ──► Recommendations
              (CNN)                         (filtering)               (Gaussian process)      (simulated annealing)
```

| Step | Approach | Status |
|---|---|---|
| 1. Condition from photos | CNN trained from scratch to predict condition from listing photos | Not started |
| 2. Hard constraints | Removes listings not on the shopping list, over budget, too far to fit in the buyer's pickup trip, or out of reach of their transport modes | **Done** |
| 3. Matching model | Gaussian process trained on buyers' ratings of listings (a neural network comparison if time allows) | Mock training data done |
| 4. Combination search | Simulated annealing picks one listing per item, trading off match score against price and trip distance with buyer-controlled sliders | **Done** |

All AI methods are implemented from scratch in Python, NumPy, and PyTorch, with no pretrained models. See the **[project proposal](docs/proposal.md)** for the full methods, evaluation plan, and related work.

### What works so far
- **Onboarding app:** a web form where the buyer enters their budget, transport modes, home address, pickup-trip limit, shopping list, and how much condition, seller trust, and brand matter to them (1 to 5). Every question is required.
- **Hard constraints:** the buyer's address is converted to coordinates with the free [US Census geocoder](https://geocoding.geo.census.gov/geocoder/). A listing is kept only if it is on the shopping list, costs no more than the budget, going there and back from home fits in the trip limit, and its one-way distance is within reach of the buyer's transport (see below). The app then shows how many listings are left for each item.
- **Combination search:** once every item has at least one listing left, the app finds the best listing for each item (simulated annealing) and shows the total price, the pickup trip length, each chosen listing, and a suggested pickup order.
- **Price and distance sliders:** two whole-number sliders (0 to 10) let the buyer choose how much total price and trip distance matter. Moving a slider reruns the search.
- **Mock data:** simulated buyers, listings with realistic titles, and buyer ratings of listings for training the matching model.

### Transport limits

The buyer can use any of the modes they pick, so the farthest they can go one way is the limit of the most generous mode selected (for example, walk + transit gives 10 miles).

| Mode | Max one-way distance |
|---|---|
| Walk | 1.5 miles |
| Transit | 10 miles |
| Car | No cap of its own; the buyer's trip limit already bounds it |

The limits are set in `MODE_MAX_MILES` in `constraints.py`. Distances are straight-line, not road or transit distances.

### How the combination search works

The search picks one listing per item (a state is one listing number per item) and minimizes an energy:

```
energy = -(total match) + n × (price_weight × price_share + trip_weight × trip_share)
```

where `n` is the number of items on the shopping list.

- **Match:** each listing's match score is rescaled to 0 to 1 across the candidates. The scores currently come from the simulated buyers' true preferences (`true_score` in `mock_buyers.py`) and will be replaced by the trained matching model.
- **Price:** `price_share` runs from 0 (cheapest possible total) to 1 (priciest possible total).
- **Trip:** `trip_share` runs from about 0 (shortest possible trip) to 1 (right at the trip limit). The trip is the shortest loop from home to every pickup and back, found by trying every pickup order.
- **Why scaled:** putting the three terms on a similar 0 to 1 range makes the sliders visibly change the result, and multiplying price and trip by `n` keeps total match equivalent to average match, so slider values mean the same thing for any shopping list length.
- **Hard limits:** the total price must stay within the budget and the full loop within the trip limit. A combination that breaks either is never taken. These use the real budget and trip limit, not the scaled values.
- **Algorithm:** first-choice hill climbing starts from the cheapest listing for each item and finds a combination that satisfies both limits. Simulated annealing then improves it, accepting worse neighbors with probability e^((current − neighbor) / T) and cooling T by a constant factor each step. The starting temperature is set so about 80% of sampled worse moves are accepted at first.
- **Sliders:** the slider value divided by 5 is the weight, so 5 (the default) is a weight of 1, 10 is a weight of 2, and 0 ignores that term. The search is seeded, so the same slider settings give the same answer.

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
| `--questions-per-buyer` | 40 | Number of listings each buyer rates |
| `--consistency` | 1.5 | How closely each buyer's ratings follow their true preferences |
| `--cold-start-share` | 0.2 | Share of buyers held out entirely |
| `--val-share` | 0.1 | Share of ratings used for validation |
| `--test-share` | 0.2 | Share of ratings used for the final test |
| `--seed` | 0 | Random seed |

**Run the app:**

```
python -m streamlit run app.py
```

This opens the app in your browser. The mock listings are scattered within 15 miles of Philadelphia, so use an address in that area, for example `1400 John F Kennedy Blvd, Philadelphia, PA` with a budget of 300, a trip limit of 16 miles, a shopping list of desk, sofa, and tv, and car selected as a transport mode (walk-only buyers only see listings within 1.5 miles, so they may have few or no matches). After you click "Find listings", the app shows the best combination, and the price and trip distance sliders below the listing counts change which combination it picks.

Model training and the evaluation scripts will be added here as they are built.

## 4. Code Organization

| File | What it does |
|---|---|
| `app.py` | Streamlit app: onboarding form, the price and trip distance sliders, and the recommended combination. |
| `constraints.py` | Step 2: address lookup, straight-line distances, and the hard-constraint filter (shopping list, budget, trip limit, transport reach). |
| `search.py` | Step 4: builds the search problem, finds the shortest pickup loop, and runs hill climbing and simulated annealing to pick one listing per item. |
| `schema.py` | Defines the `Listing` and `Buyer` formats and turns a listing into the feature vector the matching model uses. Every data source (mock now, real data later) must produce this format. |
| `mock_listings.py` | Mock listings with realistic titles, prices, seller details, and locations. To be replaced by real listings later. |
| `mock_buyers.py` | Simulated buyers: their onboarding answers, their true preferences, and their ratings of listings. |
| `generate_mock_data.py` | Generates buyers, listings, and ratings, splits them for training and evaluation, and saves everything to `data/`. |
| `requirements.txt` | Python packages to install. |
| `docs/proposal.md` | Full project proposal. |

### Generated data (`data/`)

| File | Contents | Used by |
|---|---|---|
| `listings.json` | Listing records | Constraints, model, and search |
| `buyers.json` | Simulated onboarding answers: priorities, transport modes, budget, shopping list | Model and search |
| `comparisons.csv` | Every rating a simulated buyer gives a listing, with a `split` column: `train` (fit the model), `val` (tune settings), `test` (final evaluation, used once), `cold_start` (buyers held out entirely) | Model training and evaluation |
| `true_weights.json` | Each simulated buyer's true preferences (the answer key) | **Evaluation only**, never training |

### Listing features in the match score

Each buyer's true weight for a feature is a base weight scaled by their rating (1 to 5) for the priority that controls it:

| Priority | Features |
|---|---|
| Condition | `condition`, `mentions_damage` |
| Seller trust | `seller_negative_feedback`, `seller_review_count`, `business_seller`, `returns_accepted` |
| Brand | `brand_budget`, `brand_mid`, `brand_premium` (compared with unbranded) |

Price and distance are not part of the match score; they are handled by the hard constraints and the combination search.