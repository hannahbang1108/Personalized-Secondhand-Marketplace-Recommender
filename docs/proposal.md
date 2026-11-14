# Personalized Secondhand Marketplace Recommender

**CS 5100 – Project Proposal** · Hannah Bang, Manavkrishna Manoj, Tarun Sethi

## 1. Motivation & Problem Statement

Buying secondhand on marketplaces like eBay, Facebook Marketplace, and Craigslist saves money and reduces waste, but it is tedious and full of compromises. Listings are hard to compare, photos may not match the stated condition, and the best deal depends on the buyer: someone who only walks needs nearby pickups, while someone with a car on a tight budget may drive farther to save money. Buyers furnishing a home also need several items at once, so the best single listing for each item does not always add up to the best overall purchase within their budget. We propose a system that matches listings to each buyer's preferences, checks item condition directly from listing photos, and then finds the best combination of listings for a shopping list, with sliders that let the buyer control how much price and distance matter.

## 2. Proposed Approach

The system runs in four steps. **(1) Condition from photos:** a convolutional neural network, trained from scratch, predicts each listing's condition (like new, good, or fair) from its photos, using the seller-selected condition on eBay listings as training labels. **(2) Hard constraints:** before matching, listings the buyer cannot use are removed, such as items above their budget or beyond the reach of their transport mode (for example, a listing 30 miles away for a buyer who only walks). **(3) Matching model:** a model trained across many users scores each remaining listing for a given buyer from the listing's features (predicted condition, seller rating, reviews, category, description text) and the buyer's preference features. Price and distance are deliberately left out of this score. The model is trained on pairwise "which would you rather buy?" choices. **(4) Combination search:** local search finds one listing per item on the buyer's shopping list that maximizes total match score minus slider-weighted total price and distance, while keeping the combined cost within budget. Moving a slider reruns the search. Since scraping marketplaces can break their terms of service, listing data and photos will come from eBay's Browse API, supplemented by LLM-generated mock listings grounded in the Mercari Price Suggestion dataset. We will test with simulated buyers whose true preferences we set in advance, plus a small number of real users. We will evaluate held-out pairwise accuracy of the matching model, condition accuracy of the CNN against a baseline, how close local search gets to the exact optimum on small shopping lists, and recommendation quality against a lowest-price baseline. All AI methods will be implemented from scratch in Python, NumPy, and PyTorch, with no pretrained models.

| Component | Primary approach | Comparison |
|---|---|---|
| Condition from photos | CNN trained from scratch | Logistic regression on color and texture features; seller-stated condition |
| Hard constraints | Rule-based filtering (unary constraints) | Soft penalties in the search objective instead of filtering |
| Matching model | Pairwise logistic regression with user features (Bradley-Terry) | Bayesian logistic regression; Siamese neural network (RankNet-style) |
| Combination search | Simulated annealing | Hill climbing with random restarts; exhaustive search on small lists as the optimal reference |

## 3. Existing Approaches

eBay's Best Match ranks listings using factors like relevance, price, and seller track record [1], but those factors are weighted by eBay rather than by the buyer, it ranks single items rather than combinations, and it relies on the seller's stated condition. Salimans et al. [2] learn preferences across many users from pairwise comparisons, so a new user's tastes can be estimated from only a few answers, but their work targets general product recommendation rather than marketplace listings with budgets and pickup limits. RankNet [3] trains a shared neural scoring function on pairs of items, which is the basis for our Siamese comparison model. Simulated annealing [4] is a standard local search method for large combinatorial problems like choosing one listing per item. Our project combines these ideas into one system for secondhand shopping.

## References

1. eBay, "Optimizing your listings for Best Match." <https://www.ebay.com/help/selling/listings/listing-tips/optimising-listings-best-match?id=4166>
2. T. Salimans, U. Paquet, and T. Graepel, "Collaborative learning of preference rankings," *ACM RecSys*, 2012.
3. C. Burges et al., "Learning to rank using gradient descent," *ICML*, 2005.
4. S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi, "Optimization by simulated annealing," *Science*, 220(4598), 1983.
