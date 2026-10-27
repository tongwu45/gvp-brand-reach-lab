"""Interactive demo:  streamlit run app.py"""

from dataclasses import replace

import pandas as pd
import streamlit as st

from gvplab import CampaignConfig, MarketConfig, build_world, lift_study, private_reach_report, run_campaign, summarize
from gvplab.brand_safety import KeywordClassifier, NoFilter, OracleClassifier, evaluate

st.set_page_config(page_title="GVP Brand Reach Lab", layout="wide")
st.title("GVP Brand Reach Lab")
st.caption("Simulated sandbox: extend a YouTube brand campaign to Google Video Partners inventory and "
           "see what happens to reach, cost, brand safety and measurement. All data is synthetic.")


@st.cache_resource(show_spinner="Simulating users, placements and auctions...")
def get_world(n_users: int, seed: int):
    return build_world(MarketConfig(n_users=n_users), seed=seed)


CLASSIFIERS = {
    "Context-aware (oracle upper bound)": OracleClassifier(),
    "Keyword blocklist": KeywordClassifier(),
    "No filter": NoFilter(),
}

with st.sidebar:
    st.header("Market")
    n_users = st.select_slider("Users", [20_000, 50_000, 100_000, 200_000], value=50_000)
    seed = st.number_input("Seed", 0, 99, 0)
    st.header("Campaign")
    budget = st.slider("Budget ($)", 50, 2_000, 200, step=50)
    cap = st.slider("Frequency cap", 1, 10, 4)
    gvp_on = st.toggle("Extend to GVP inventory", value=True)
    suitability = st.radio("Brand suitability", ["limited", "standard", "expanded"], index=1, horizontal=True)
    clf_name = st.selectbox("Brand-safety classifier", list(CLASSIFIERS))
    auction = st.selectbox("Auction", ["second_price / truthful", "first_price / truthful", "first_price / shaded"])
    st.header("Measurement")
    eps = st.select_slider("Privacy epsilon", [0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)
    k = st.select_slider("k-threshold", [0, 10, 25, 50, 100], value=10)
    dims = st.radio("Report granularity", ["age x region", "dma x age"], horizontal=True)

world = get_world(n_users, seed)
# Keep the budget meaningful relative to market size.
scale = n_users / 200_000
a, b = auction.split(" / ")
cfg = CampaignConfig(budget=budget * scale * 4, frequency_cap=cap, suitability=suitability,
                     auction=a, bidding=b, networks=("youtube", "gvp") if gvp_on else ("youtube",))
pred = CLASSIFIERS[clf_name].predict(world.placements)

d = run_campaign(world, cfg, pred, seed=seed)
s = summarize(d, world)
yt = summarize(run_campaign(world, replace(cfg, networks=("youtube",)), OracleClassifier().predict(world.placements),
                            seed=seed), world)

c = st.columns(5)
c[0].metric("Unique reach", f"{s['reach_pct']:.1%}", f"{s['reach_pct'] / yt['reach_pct'] - 1:+.1%} vs YouTube-only")
c[1].metric("vCPM", f"${s['vcpm']:.2f}", f"{s['vcpm'] / yt['vcpm'] - 1:+.1%}", delta_color="inverse")
c[2].metric("Cost per reached user", f"${s['cost_per_1k_reached']:.2f} / 1k users",
            f"{s['cost_per_1k_reached'] / yt['cost_per_1k_reached'] - 1:+.1%}", delta_color="inverse")
c[3].metric("Unsafe impressions", f"{s['unsafe_impression_rate']:.2%}")
c[4].metric("Reach only via GVP", f"{s['gvp_only_reach']:,}")

left, right = st.columns(2)
with left:
    st.subheader("Delivery by supply")
    imp = d.impressions.assign(pub_tier=world.placements.pub_tier.values[d.impressions.placement_id.values])
    by = imp.groupby("pub_tier").agg(impressions=("user_id", "size"), viewable=("viewable", "mean"),
                                     spend=("price_cpm", lambda x: x.sum() / 1000),
                                     avg_cpm=("price_cpm", "mean"))
    st.dataframe(by.style.format({"viewable": "{:.0%}", "spend": "${:,.0f}", "avg_cpm": "${:.2f}"}))

    st.subheader("Brand-safety classifier quality")
    ev = pd.DataFrame({n: evaluate(clf.predict(world.placements), world.placements, suitability)
                       for n, clf in CLASSIFIERS.items()}).T
    st.dataframe(ev.style.format("{:.1%}"))
    st.caption("over_block = suitable traffic wrongly blocked (lost publisher revenue); "
               "leak = unsuitable traffic let through (advertiser risk).")

with right:
    st.subheader("Privacy-safe reach report")
    cells, ps = private_reach_report(d, world, epsilon=eps, k=k, dims=tuple(dims.split(" x ")), seed=seed)
    m = st.columns(3)
    m[0].metric("Total reach error", f"{ps['total_reach_error']:.2%}")
    m[1].metric("GVP-only reach error", f"{ps['incremental_gvp_reach_error']:.2%}")
    m[2].metric("Reach in suppressed cells", f"{ps['suppressed_reach_share']:.1%}")
    st.dataframe(cells.head(30), height=240)

    st.subheader("Brand-lift study (holdout)")
    lift = lift_study(d, world, seed=seed)
    st.write(f"Estimated lift **{lift['lift_estimate']:+.2%}** "
             f"(95% CI {lift['ci_low']:+.2%} to {lift['ci_high']:+.2%}); simulated true lift {lift['true_lift']:+.2%}. "
             f"{'Significant.' if lift['significant'] else 'Not significant: the study is underpowered.'}")
    with st.expander("Sample of placements and their labels"):
        sample = world.placements.sample(20, random_state=seed).assign(predicted=lambda x: pred[x.placement_id])
        st.dataframe(sample[["network", "pub_tier", "title", "true_tier", "predicted", "viewability"]])
