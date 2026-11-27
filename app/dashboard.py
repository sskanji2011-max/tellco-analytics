"""TellCo dashboard.  Run:  streamlit run app/dashboard.py   (after `tellco-run --input telcom_data.xlsx`)"""
import os
import pandas as pd
import plotly.express as px
import streamlit as st

DATA = os.environ.get("TELLCO_DATA", "data/processed")
D = os.path.join(DATA, "dashboard")
st.set_page_config(page_title="TellCo Analytics", layout="wide")


@st.cache_data
def read(name, **kw):
    return pd.read_csv(os.path.join(D, name), **kw)


@st.cache_data
def features_table():
    return pd.read_csv(os.path.join(DATA, "feature_store", "features.csv"), index_col=0)


if not os.path.exists(os.path.join(DATA, "feature_store", "features.csv")):
    st.error("No processed data found. Run `tellco-run --input telcom_data.xlsx` first.")
    st.stop()

feat = features_table()
st.title("TellCo user analytics")
page = st.sidebar.radio("Section", ["Summary", "1 · User overview", "2 · Engagement", "3 · Experience",
                                    "4 · Satisfaction", "5 · Extra analysis", "Verdict"])

if page == "Summary":
    c = st.columns(4)
    c[0].metric("Customers", f"{len(feat):,}")
    c[1].metric("Avg data / customer", f"{feat['Total Data'].mean() / 1e6:,.0f} MB")
    c[2].metric("Good experience", f"{(feat['Experience Group'] == 'Good experience').mean():.0%}")
    c[3].metric("Less satisfied", f"{(feat['Segment'] == 'Less satisfied').mean():.0%}")
    seg = feat.groupby("Segment")[["Engagement Score", "Experience Score", "Satisfaction Score"]].mean()
    st.subheader("Average scores by satisfaction segment")
    st.dataframe(seg.style.format("{:.3f}"))

elif page == "1 · User overview":
    c1, c2 = st.columns(2)
    th = read("top_handsets.csv", index_col=0).reset_index()
    th.columns = ["Handset", "Sessions"]
    c1.plotly_chart(px.bar(th.iloc[::-1], x="Sessions", y="Handset", orientation="h", title="Top 10 handsets"),
                    width="stretch")
    tm = read("top_manufacturers.csv", index_col=0).reset_index()
    tm.columns = ["Manufacturer", "Sessions"]
    c2.plotly_chart(px.bar(tm, x="Manufacturer", y="Sessions", title="Top 3 manufacturers"), width="stretch")
    per = read("top_handsets_per_manufacturer.csv")
    m = st.selectbox("Manufacturer", per["manufacturer"].unique())
    st.plotly_chart(px.bar(per[per["manufacturer"] == m].iloc[::-1], x="sessions", y="handset", orientation="h",
                           title=f"Top 5 {m} handsets"), width="stretch")
    apps = read("app_totals.csv", index_col=0).reset_index()
    apps.columns = ["App", "Bytes"]
    apps["TB"] = apps["Bytes"] / 1e12
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.bar(apps.sort_values("TB", ascending=False), x="App", y="TB", title="Data by application (TB)"),
                    width="stretch")
    dec = read("deciles.csv")
    dec["TB"] = dec["total_data"] / 1e12
    c4.plotly_chart(px.bar(dec, x="Decile", y="TB", title="Total data by duration decile (1 = shortest)"),
                    width="stretch")
    corr = read("app_correlation.csv", index_col=0)
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Correlation of application volumes"), width="stretch")
    pca = read("pca.csv")
    pca["cumulative"] = pca["explained"].cumsum()
    st.plotly_chart(px.line(pca, x="component", y=["explained", "cumulative"], markers=True, title="PCA"),
                    width="stretch")

elif page == "2 · Engagement":
    cl = read("engagement_clusters.csv")
    st.dataframe(cl.style.format({"sessions": "{:.1f}", "duration_h": "{:.1f}", "data_GB": "{:.2f}", "total_TB": "{:.1f}"}))
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(cl, x="Persona", y="users", title="Users per engagement cluster"), width="stretch")
    c2.plotly_chart(px.bar(cl, x="Persona", y="total_TB", title="Total traffic (TB) per cluster"), width="stretch")
    st.plotly_chart(px.line(read("elbow.csv"), x="k", y="inertia", markers=True, title="Elbow method"),
                    width="stretch")
    metric = st.selectbox("Top 10 customers by", ["Sessions", "Duration_ms", "Total Data"])
    st.dataframe(feat[metric].nlargest(10).to_frame())

elif page == "3 · Experience":
    st.dataframe(read("experience_clusters.csv"))
    he = read("handset_experience.csv")
    st.plotly_chart(px.scatter(he, x="avg_rtt", y="avg_throughput", size="users", color="avg_tcp_MB", hover_name="Handset Type",
                               title="Handsets: RTT vs throughput (bubble = customers, colour = TCP retrans MB)"),
                    width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(he.nlargest(10, "avg_throughput"), x="Handset Type", y="avg_throughput", title="Best throughput (kbps)"),
                    width="stretch")
    c2.plotly_chart(px.bar(he.nlargest(10, "avg_tcp_MB"), x="Handset Type", y="avg_tcp_MB", title="Highest TCP retransmission (MB)"),
                    width="stretch")

elif page == "4 · Satisfaction":
    st.dataframe(read("segments.csv", index_col=0).style.format("{:.3f}"))
    sample = feat.sample(min(20000, len(feat)), random_state=1)
    st.plotly_chart(px.scatter(sample, x="Engagement Score", y="Experience Score", color="Segment", opacity=0.5,
                               title="Satisfaction segments (k = 2)"), width="stretch")
    st.subheader("Top 10 satisfied customers")
    st.dataframe(read("top10_satisfied.csv"))
    path = os.path.join(DATA, "model_comparison.csv")
    if os.path.exists(path):
        st.subheader("Regression models (held-out 20%)")
        st.dataframe(pd.read_csv(path, index_col=0).style.format("{:.4f}"))
    st.download_button("Download final scores (CSV)", open(os.path.join(DATA, "final_scores.csv"), "rb"),
                       file_name="final_scores.csv")

elif page == "5 · Extra analysis":
    def extra(name, **kw):
        path = os.path.join(D, name)
        return pd.read_csv(path, **kw) if os.path.exists(path) else None

    summ = extra("extra_summary.csv", index_col=0)
    if summ is None:
        st.info("Extra-analysis files not found. Run Part 5B of the notebook: it writes data/processed/dashboard/extra_*.csv.")
        st.stop()
    s = summ["value"]
    c = st.columns(4)
    c[0].metric("Gini of data usage", f"{s['gini']:.2f}")
    c[1].metric("Data used by top 10% of customers", f"{s['top10_share']:.1f}%")
    c[2].metric("Traffic in weak-experience groups", f"{s['weak_traffic_share']:.1f}%")
    c[3].metric("Brand vs experience (Cramér's V)", f"{s['cramers_v_brand']:.2f}")

    st.subheader("Concentration of usage")
    lz = extra("extra_lorenz.csv")
    fig = px.line(lz, x="x", y="y", title="Lorenz curve of total data per customer",
                  labels={"x": "Cumulative share of customers", "y": "Cumulative share of data"})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("How precise are the headline figures? (bootstrap 95% intervals)")
    ci = extra("extra_ci.csv", index_col=0)
    st.dataframe(ci.style.format("{:.2f}"))

    st.subheader("Handset brand and experience")
    be = extra("extra_brand_experience.csv", index_col=0)
    long = be.reset_index().melt(id_vars="brand", var_name="Experience group", value_name="Share of brand's customers (%)")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(long, x="brand", y="Share of brand's customers (%)", color="Experience group", barmode="stack",
                           title="Experience group by handset brand"), width="stretch")
    pe = extra("extra_persona_experience.csv", index_col=0)
    c2.plotly_chart(px.imshow(pe, text_auto=".1f", color_continuous_scale="Blues",
                              title="Experience group by engagement persona (% of persona)"), width="stretch")
    st.caption("Effect sizes (rank-biserial r) from Mann-Whitney tests on average throughput; 0 = no difference.")
    st.dataframe(extra("extra_brand_tests.csv", index_col=0).style.format({"Median A (kbps)": "{:,.0f}", "Median B (kbps)": "{:,.0f}", "Rank-biserial r": "{:.2f}"}))

    st.subheader("When do sessions happen?")
    hr, wk = extra("extra_hourly.csv"), extra("extra_weekday.csv")
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.bar(hr, x="hour", y="pct_sessions", title="Share of sessions by hour of day (%)"), width="stretch")
    c4.plotly_chart(px.bar(wk, x="day", y="TB_per_day", title="Average data per day with data (TB)"), width="stretch")

    st.subheader("Model drivers and segment stability")
    imp, stab = extra("extra_importance.csv"), extra("extra_stability.csv", index_col=0)
    c5, c6 = st.columns(2)
    c5.plotly_chart(px.bar(imp.iloc[::-1], x="r2_lost", y="feature", orientation="h",
                           title="Permutation importance (R2 lost when shuffled)"), width="stretch")
    c6.markdown("**Adjusted Rand index across eight random starts** (1.0 = identical groups)")
    c6.dataframe(stab.style.format("{:.3f}"))

else:
    st.subheader("Recommendation: conditional buy")
    st.markdown("""
- Large, active base with real upside from fixing network quality: fixed-wireless devices show the network can deliver about 10 Mbps.
- Only about a third of customers have a good experience, and weak experience reaches about two thirds of traffic (see 5 · Extra analysis); satisfaction tracks experience far more than usage.
- Buy only if the price leaves room for network investment. Revenue, ARPU and churn are not in this dataset, so the final valuation needs them.
""")
