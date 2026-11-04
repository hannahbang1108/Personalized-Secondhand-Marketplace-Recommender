# Personalized Secondhand Marketplace Recommender

**CS 5100 Project** · Hannah Bang, Manavkrishna Manoj, Tarun Sethi

A recommender that matches secondhand listings to each buyer's preferences, checks item condition directly from listing photos, and finds the best combination of listings for a whole shopping list, with sliders that let the buyer control how much price and distance matter.

---

## Motivation

Buying secondhand on marketplaces like eBay, Facebook Marketplace, and Craigslist saves money and reduces waste, but it is tedious and full of compromises:

- **Listings are hard to compare**, and photos may not match the stated condition.
- **The best deal depends on the buyer.** Someone who only walks needs nearby pickups, while someone with a car on a tight budget may drive farther to save money.
- **Buyers often need several items at once.** When furnishing a home, the best single listing for each item does not always add up to the best overall purchase within budget.

## How It Works

The system runs as a four-stage pipeline:

```
Listings ──► 1. Condition from photos ──► 2. Hard constraints ──► 3. Matching model ──► 4. Combination search ──► Recommendations
              (CNN)                         (filtering)               (pairwise ranking)      (simulated annealing)
```

1. **Condition from photos.** A convolutional neural network, trained from scratch, predicts each listing's condition (*like new*, *good*, or *fair*) from its photos. Seller-selected conditions on eBay listings serve as training labels.
2. **Hard constraints.** Listings the buyer cannot use are removed before matching, such as items above their budget or beyond the reach of their transport mode (e.g., a listing 30 miles away for a buyer who only walks).
3. **Matching model.** A model trained across many users scores each remaining listing for a given buyer, using listing features (predicted condition, seller rating, reviews, category, description text) and the buyer's preference features. It is trained on pairwise *"which would you rather buy?"* choices. Price and distance are deliberately left out of this score.
4. **Combination search.** Local search picks one listing per item on the shopping list to maximize

   ```
   total match score − w_price · total price − w_distance · total distance
   ```

   subject to the combined cost staying within budget. The weights come from the buyer's sliders, and moving a slider reruns the search.

## Methods

| Component | Primary approach | Comparison |
|---|---|---|
| Condition from photos | CNN trained from scratch | Logistic regression on color and texture features; seller-stated condition |
| Hard constraints | Rule-based filtering (unary constraints) | Soft penalties in the search objective instead of filtering |
| Matching model | Pairwise logistic regression with user features (Bradley-Terry) | Bayesian logistic regression; Siamese neural network (RankNet-style) |
| Combination search | Simulated annealing | Hill climbing with random restarts; exhaustive search on small lists as the optimal reference |

All AI methods are implemented **from scratch** in Python, NumPy, and PyTorch, with **no pretrained models**.

## Data

Because scraping marketplaces can violate their terms of service, data comes from:

- **[eBay Browse API](https://developer.ebay.com/api-docs/buy/browse/overview.html)** for listing data and photos
- **LLM-generated mock listings** grounded in the [Mercari Price Suggestion dataset](https://www.kaggle.com/c/mercari-price-suggestion-challenge)

## Evaluation

We test with **simulated buyers** whose true preferences are set in advance, plus a small number of **real users**.

| Component | Metric |
|---|---|
| Matching model | Held-out pairwise accuracy |
| Condition CNN | Condition accuracy against a baseline |
| Combination search | Gap to the exact optimum on small shopping lists |
| Full system | Recommendation quality against a lowest-price baseline |

## Related Work

- **eBay Best Match** [1] ranks listings by factors like relevance, price, and seller track record. However, those factors are weighted by eBay rather than the buyer, it ranks single items rather than combinations, and it relies on the seller's stated condition.
- **Salimans et al.** [2] learn preferences across many users from pairwise comparisons, so a new user's tastes can be estimated from only a few answers. Their work targets general product recommendation rather than marketplace listings with budgets and pickup limits.
- **RankNet** [3] trains a shared neural scoring function on pairs of items and is the basis for our Siamese comparison model.
- **Simulated annealing** [4] is a standard local search method for large combinatorial problems like choosing one listing per item.

This project combines these ideas into one system for secondhand shopping.

## References

1. eBay, "Optimizing your listings for Best Match." <https://www.ebay.com/help/selling/listings/listing-tips/optimising-listings-best-match?id=4166>
2. T. Salimans, U. Paquet, and T. Graepel, "Collaborative learning of preference rankings," *ACM RecSys*, 2012.
3. C. Burges et al., "Learning to rank using gradient descent," *ICML*, 2005.
4. S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi, "Optimization by simulated annealing," *Science*, 220(4598), 1983.
