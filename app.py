# Run with: python -m streamlit run app.py
import random
import urllib.error

import streamlit as st

from constraints import geocode, filter_listings, load_listings, miles_between
from schema import Buyer, ITEMS_BY_CATEGORY, TRANSPORT_MODES, PRIORITIES
from search import make_problem, shortest_distance, solve

ALL_ITEMS = {item: category
             for category, items in ITEMS_BY_CATEGORY.items()
             for item in items}

st.set_page_config(page_title="Secondhand Recommender")


# Loads the listings once and reuses them on every rerun of the page.
@st.cache_data
def get_listings():
    return load_listings()


st.title("Secondhand Marketplace Recommender")

with st.form("onboarding"):
    name = st.text_input("Your name").strip()
    budget = st.number_input("Total budget for everything ($)", min_value=1.0,
                             value=None, step=10.0)
    transport = st.multiselect("Which modes of transportation can you use?",
                               TRANSPORT_MODES)
    address = st.text_input("Home address", placeholder="360 Huntington Ave, Boston, MA",
                            help="Sent to the US Census geocoder to find your location.").strip()
    trip_miles = st.number_input("Most miles you'd travel in total for one pickup trip "
                                 "(leaving from and returning home)",
                                 min_value=0.1, value=None)
    items = st.multiselect("What's on your shopping list?", list(ALL_ITEMS),
                           format_func=lambda i: f"{i} ({ALL_ITEMS[i]})")

    # used by the matching model (step 3)
    st.markdown("**How much does each of these matter to you?** "
                "1 = doesn't matter, 5 = matters a lot")
    priorities = {p: st.radio(p.replace("_", " ").capitalize(), [1, 2, 3, 4, 5],
                              index=None, horizontal=True)
                  for p in PRIORITIES}

    submitted = st.form_submit_button("Find listings", type="primary")

if submitted:
    errors = []
    if not name:
        errors.append("Please enter your name.")
    if budget is None:
        errors.append("Please enter your budget.")
    if not transport:
        errors.append("Please pick at least one mode of transportation.")
    if not address:
        errors.append("Please enter your address.")
    if trip_miles is None:
        errors.append("Please enter a trip limit.")
    if not items:
        errors.append("Please pick at least one item.")
    if None in priorities.values():
        errors.append("Please rate all three priorities.")

    match = None
    if address:
        try:
            match = geocode(address)
            if match is None:
                errors.append("Couldn't find that address. Check the spelling "
                              "and include the city and state.")
        except (urllib.error.URLError, TimeoutError, ValueError):
            errors.append("Couldn't reach the address lookup. Please try again.")

    if errors:
        st.session_state.pop("results", None)
        for error in errors:
            st.error(error)
    else:
        matched_address, latitude, longitude = match
        buyer = Buyer(buyer_id="U_live", name=name, priorities=priorities,
                      transport=transport, budget=budget,
                      shopping_list=[{"item": i, "category": ALL_ITEMS[i]} for i in items],
                      home_latitude=latitude, home_longitude=longitude,
                      trip_miles=trip_miles)
        candidates = filter_listings(buyer, (latitude, longitude), trip_miles,
                                     get_listings())
        # the search needs at least one listing for every item
        can_search = all(candidates.values())
        st.session_state.results = {
            "buyer": buyer,
            "matched_address": matched_address,
            "trip_miles": trip_miles,
            "candidates": candidates,
            # built once per submit so the route cache survives slider changes
            "problem": (make_problem(buyer, (latitude, longitude), trip_miles, candidates)
                        if can_search else None),
        }

if "results" in st.session_state:
    results = st.session_state.results
    buyer = results["buyer"]
    st.header(f"Hi {buyer.name}!")
    st.caption(f"Pickups from {results['matched_address']} · "
               f"Budget ${buyer.budget:,.2f} · Trip limit {results['trip_miles']:g} miles")

    st.subheader("Listings that fit your limits")
    for item, options in results["candidates"].items():
        st.write(f"**{item}**: {len(options)} listings")

    # the search needs at least one listing for every item
    missing = [item for item, options in results["candidates"].items() if not options]
    if missing:
        st.warning(f"No listings left for: {', '.join(missing)}. "
                   "Try raising your trip limit or your budget.")
    else:
        problem = results["problem"]
        home = (buyer.home_latitude, buyer.home_longitude)

        st.subheader("What should the search favor?")
        # sliders live outside the form, so moving one reruns the page right away
        # whole-number 0-10 sliders; dividing by 5 gives weights from 0 to 2 (5 = weight 1.0)
        price_importance = st.slider(
            "Price", 0, 10, 5,
            help="Higher = the search works harder to keep the total price down.")
        trip_importance = st.slider(
            "Trip distance", 0, 10, 5,
            help="Higher = the search works harder to keep the pickup loop short.")
        problem["price_weight"] = price_importance / 5
        problem["trip_weight"] = trip_importance / 5

        # the search is random; seeding makes the same slider values give the same answer
        random.seed(0)
        best = solve(problem)

        if best is None:
            st.error("Couldn't find a combination that fits both your budget "
                     "and your trip limit. Try raising either one.")
        else:
            miles, order = shortest_distance(problem, best)
            total_price = sum(problem["prices"][i][j] for i, j in enumerate(best))

            st.subheader("Your best combination")
            col1, col2 = st.columns(2)
            col1.metric("Total price", f"${total_price:,.2f}",
                        f"${buyer.budget - total_price:,.2f} under budget",
                        delta_color="off")
            col2.metric("Pickup trip", f"{miles:.1f} mi",
                        f"{results['trip_miles'] - miles:.1f} mi under limit",
                        delta_color="off")

            for item_num, listing_num in enumerate(best):
                listing = problem["listings_by_item"][item_num][listing_num]
                away = miles_between(*home, listing.latitude, listing.longitude)
                with st.container(border=True):
                    st.markdown(f"**{problem['item_names'][item_num].capitalize()}:** "
                                f"{listing.title}")
                    st.write(f"${listing.price:,.2f} · "
                             f"{listing.condition.replace('_', ' ')} · "
                             f"{away:.1f} mi from home")
                    st.caption(f"Seller {listing.seller_feedback_percent:g}% positive "
                               f"({listing.seller_review_count:,} reviews) · "
                               f"{'returns accepted' if listing.returns_accepted else 'no returns'} · "
                               f"match score "
                               f"{problem['match_scores'][item_num][listing_num]:.2f}")

            route = " → ".join(["Home"] + [problem["item_names"][i] for i in order] + ["Home"])
            st.write(f"**Suggested pickup order:** {route}")