"""
FocusFlash - A virtual consumer focus group in 60 seconds.

Pipeline:
1. Real customer data (Kaggle) -> K-Means segmentation (scikit-learn)
2. Each segment gets a meaningful name computed from its profile
3. Each segment becomes a persona played by an LLM (Mistral AI)
4. Generated reviews are checked by an independent sentiment model (Hugging Face)
5. The LLM summarises the results into marketing recommendations
Features: single-product panel, A/B test of two versions, one-to-one interview with a persona.
"""

import html
import json
import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import pipeline

# ---------- Settings ----------
DATA_PATH = Path(__file__).parent / "data" / "marketing_campaign.csv"
KAGGLE_DATASET = "imakash3011/customer-personality-analysis"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "open-mistral-nemo"  # served by Mistral as ministral-8b
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
REF_YEAR = 2014  # approximate year the data was collected
FEATURES = ["Age", "Income", "Children", "Spending", "WebShare",
            "PromoShare", "MntSweetProducts", "MntWines"]
COLORS = ["#5B3DF5", "#F59E0B", "#10B981", "#EF4444", "#0EA5E9", "#EC4899"]
QUICK_QUESTIONS = ["What would make you change your mind?",
                   "What price would feel fair to you?",
                   "Where would you expect to find this product?"]

# ---------- Prompts ----------
SYSTEM_PERSONA = """You are playing a real consumer, a member of a focus group.
You are given the statistical profile of your customer segment and a product to test.
Invent a credible person matching this profile, then react to the product
as they really would: with their budget constraints, their habits and
their frankness. Don't be complacent: if the product doesn't suit you, say so.

Reply ONLY in JSON with these keys:
- "first_name": the person's first name
- "portrait": 2 sentences about their life and shopping habits
- "review": 3 to 4 sentences in the first person, in a natural, spoken tone
- "purchase_intent": integer from 1 (never) to 10 (definitely)
- "max_price": maximum price in euros they would pay (number)
- "likes": list of 2 items maximum
- "concerns": list of 2 items maximum"""

SYSTEM_PERSONA_AB = """You are playing a real consumer, a member of a focus group.
You are given the statistical profile of your customer segment and TWO versions
of a product (A and B). Invent a credible person matching this profile, then
react to each version as they really would: with their budget constraints,
their habits and their frankness. Don't be complacent.

Reply ONLY in JSON with these keys:
- "first_name": the person's first name
- "portrait": 2 sentences about their life and shopping habits
- "review_a": 2 to 3 sentences in the first person about version A
- "review_b": 2 to 3 sentences in the first person about version B
- "intent_a": integer from 1 (never) to 10 (definitely) for version A
- "intent_b": integer from 1 (never) to 10 (definitely) for version B
- "preference": "A" or "B"
- "reason": one sentence explaining the preference"""

SYSTEM_SUMMARY = """You are a marketing research director. You are given the reactions
of a panel of virtual consumers (one per segment, with the segment size)
to a new product. Produce an actionable summary.

Reply ONLY in JSON with these keys:
- "verdict": a one-sentence overall verdict
- "priority_segment": the segment to target first and why (1-2 sentences)
- "strengths": list of 3 strengths of the product
- "risks": list of 3 main risks or barriers
- "improvements": list of 3 concrete recommendations
- "recommended_price": recommended price in euros with a short justification
- "slogan": a suggested slogan"""

SYSTEM_SUMMARY_AB = """You are a marketing research director. You are given the reactions
of a panel of virtual consumers (one per segment, with the segment size)
to two versions of a product (A and B). Produce an actionable summary of the A/B test.

Reply ONLY in JSON with these keys:
- "verdict": a one-sentence overall verdict
- "winner": "A" or "B"
- "why": 1 to 2 sentences explaining why this version wins
- "segments_a": the segments that prefer A, in one sentence
- "segments_b": the segments that prefer B, in one sentence
- "improvements": list of 3 recommendations to improve the winning version
- "slogan": a suggested slogan for the winning version"""

SYSTEM_INTERVIEW = """You are {first_name}, a consumer taking part in a focus group.
Your portrait: {portrait}
Statistical profile of your segment ({segment}): {profile}

The product tested: {product}
What you said about it during the panel: {review}

The moderator is now asking you questions. Answer in character, in the first person,
in 2 to 4 sentences, with a natural, spoken tone. Stay consistent with your initial
opinion, unless you are given a genuinely convincing argument.
Never say that you are an AI."""

# ---------- Design ----------
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.ff-hero {background: linear-gradient(135deg, #5B3DF5 0%, #8B5CF6 60%, #EC4899 100%);
  border-radius: 20px; padding: 28px 32px; color: #fff; margin-bottom: 1.2rem;}
.ff-logo {font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;}
.ff-tagline {font-size: 1.1rem; opacity: .92; margin-top: 2px;}
.ff-steps {display: flex; gap: 12px; flex-wrap: wrap; margin-top: 18px;}
.ff-step {background: rgba(255,255,255,.16); border-radius: 12px; padding: 8px 14px;
  font-size: .9rem;}
.ff-step b {background: #fff; color: #5B3DF5; border-radius: 50%; display: inline-block;
  width: 22px; height: 22px; text-align: center; line-height: 22px; margin-right: 6px;}
.ff-card {background: #fff; border: 1px solid #E7E3FF; border-radius: 16px; padding: 18px;
  margin-bottom: 16px; box-shadow: 0 4px 14px rgba(91,61,245,.07); color: #1E1B3A;}
.ff-head {display: flex; align-items: center; gap: 12px; margin-bottom: 10px;}
.ff-avatar {width: 46px; height: 46px; border-radius: 50%; color: #fff; font-weight: 700;
  font-size: 1.2rem; display: flex; align-items: center; justify-content: center;}
.ff-name {font-weight: 700; font-size: 1.1rem;}
.ff-seg {font-size: .8rem; color: #6B6890;}
.ff-portrait {font-size: .85rem; color: #6B6890; margin-bottom: 10px;}
.ff-quote {font-style: italic; background: #F6F4FF; border-left: 4px solid #5B3DF5;
  border-radius: 8px; padding: 10px 12px; font-size: .93rem; margin-bottom: 12px;}
.ff-label {font-size: .82rem; color: #6B6890; margin: 6px 0 4px;}
.ff-gauge {background: #EEEAFB; border-radius: 99px; height: 9px; overflow: hidden;}
.ff-gauge div {height: 100%; border-radius: 99px;}
.ff-meta {font-size: .85rem; margin-top: 10px;}
.ff-tag {display: inline-block; border-radius: 99px; padding: 3px 10px; font-size: .78rem;
  margin: 6px 4px 0 0;}
.ff-plus {background: #E7F8F1; color: #0B7A55;}
.ff-minus {background: #FDECEC; color: #B42318;}
.ff-kpi {background: #fff; border: 1px solid #E7E3FF; border-radius: 16px; padding: 16px 18px;
  color: #1E1B3A; height: 100%;}
.ff-kpi-label {font-size: .82rem; color: #6B6890;}
.ff-kpi-value {font-size: 1.9rem; font-weight: 800; margin-top: 2px;}
.ff-kpi-sub {font-size: .8rem; color: #6B6890;}
.ff-box {border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; color: #1E1B3A;}
.ff-box h4 {margin: 0 0 6px; font-size: 1rem;}
.ff-box ul {margin: 0; padding-left: 18px;}
.ff-verdict {background: linear-gradient(135deg, #F1EEFF, #FCE7F3); border-radius: 16px;
  padding: 18px 20px; font-size: 1.1rem; color: #1E1B3A; margin-bottom: 16px;}
.ff-slogan {text-align: center; font-size: 1.4rem; font-weight: 700; color: #5B3DF5;
  padding: 18px; border: 2px dashed #C9BEFF; border-radius: 16px; margin-top: 8px;}
</style>
"""


def esc(text) -> str:
    return html.escape(str(text))


def gauge_color(score: float) -> str:
    return "#10B981" if score >= 7 else "#F59E0B" if score >= 4 else "#EF4444"


def stars_txt(n: int) -> str:
    return "★" * n + "☆" * (5 - n)


def kpi(label: str, value: str, sub: str = "", color: str = "#1E1B3A") -> str:
    return (f'<div class="ff-kpi"><div class="ff-kpi-label">{esc(label)}</div>'
            f'<div class="ff-kpi-value" style="color:{color}">{esc(value)}</div>'
            f'<div class="ff-kpi-sub">{esc(sub)}</div></div>')


def box(title: str, items: list, bg: str) -> str:
    lis = "".join(f"<li>{esc(x)}</li>" for x in items)
    return f'<div class="ff-box" style="background:{bg}"><h4>{esc(title)}</h4><ul>{lis}</ul></div>'


def gauge(label: str, score: int) -> str:
    return (f'<div class="ff-label">{esc(label)} <b>{score}/10</b></div>'
            f'<div class="ff-gauge"><div style="width:{score * 10}%;'
            f'background:{gauge_color(score)}"></div></div>')


def persona_card(r: dict, mode: str) -> str:
    color = COLORS[r["segment"] % len(COLORS)]
    head = (f'<div class="ff-head"><div class="ff-avatar" style="background:{color}">'
            f'{esc(r["first_name"][:1].upper())}</div><div>'
            f'<div class="ff-name">{esc(r["first_name"])}</div>'
            f'<div class="ff-seg">{esc(r["segment_name"])} · {r["size"]} customers</div></div></div>'
            f'<div class="ff-portrait">{esc(r["portrait"])}</div>')
    if mode == "single":
        tags = "".join(f'<span class="ff-tag ff-plus">👍 {esc(x)}</span>' for x in r["likes"])
        tags += "".join(f'<span class="ff-tag ff-minus">👎 {esc(x)}</span>' for x in r["concerns"])
        body = (f'<div class="ff-quote">“{esc(r["review"])}”</div>'
                + gauge("Purchase intent", r["purchase_intent"])
                + f'<div class="ff-meta">Max price: <b>€{esc(r["max_price"])}</b> · '
                  f'Hugging Face sentiment: {stars_txt(r["hf_stars"])}</div>{tags}')
    else:
        pref_color = "#5B3DF5" if r["preference"] == "A" else "#EC4899"
        body = (f'<div class="ff-quote"><b>A:</b> “{esc(r["review_a"])}”</div>'
                f'<div class="ff-quote" style="border-left-color:#EC4899"><b>B:</b> '
                f'“{esc(r["review_b"])}”</div>'
                + gauge("Intent for version A", r["intent_a"])
                + gauge("Intent for version B", r["intent_b"])
                + f'<div class="ff-meta">Prefers version <b style="color:{pref_color}">'
                  f'{esc(r["preference"])}</b>: {esc(r["reason"])}</div>')
    return f'<div class="ff-card">{head}{body}</div>'


# ---------- Data (Kaggle) ----------
@st.cache_data
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        path = DATA_PATH
    else:  # automatic download if the file is missing
        import kagglehub
        path = Path(kagglehub.dataset_download(KAGGLE_DATASET)) / "marketing_campaign.csv"

    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["Income"])
    df = df[(df["Year_Birth"] >= 1930) & (df["Income"] < 200_000)].copy()  # remove outliers

    df["Age"] = REF_YEAR - df["Year_Birth"]
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    spending = ["MntWines", "MntFruits", "MntMeatProducts",
                "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df["Spending"] = df[spending].sum(axis=1)
    purchases = df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum(axis=1).replace(0, 1)
    df["WebShare"] = df["NumWebPurchases"] / purchases
    df["PromoShare"] = (df["NumDealsPurchases"] / purchases).clip(upper=1)
    return df


# ---------- Segmentation (scikit-learn) and segment names ----------
def name_segments(profiles: pd.DataFrame, df: pd.DataFrame) -> list:
    """Gives each segment a meaningful name, by comparing its profile to all customers."""
    q1, q2 = df["Income"].quantile([0.33, 0.66])
    names = []
    for p in profiles.itertuples():
        income = "Affluent" if p.Income > q2 else "Budget" if p.Income < q1 else "Middle-income"
        family = ("no kids" if p.Children < 0.5 else
                  "parents" if p.Children < 1.5 else "large families")
        traits = []
        if p.PromoShare > df["PromoShare"].mean() * 1.3:
            traits.append("deal hunters")
        if p.WebShare > df["WebShare"].mean() * 1.15:
            traits.append("online shoppers")
        if p.Spending > df["Spending"].quantile(0.75):
            traits.append("big spenders")
        if p.Age > df["Age"].quantile(0.75):
            traits.append("seniors")
        name = f"{income} · {family}"
        if traits:
            name += f" · {traits[0]}"
        if name in names and len(traits) > 1:
            name += f" · {traits[1]}"
        base, n = name, 2
        while name in names:
            name, n = f"{base} ({n})", n + 1
        names.append(name)
    return names


@st.cache_data
def segment(df: pd.DataFrame, k: int):
    X = StandardScaler().fit_transform(df[FEATURES])
    df = df.copy()
    df["Segment"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)

    profiles = df.groupby("Segment").agg(
        Size=("ID", "count"),
        Age=("Age", "median"),
        Income=("Income", "median"),
        Children=("Children", "mean"),
        Spending=("Spending", "median"),
        WebShare=("WebShare", "mean"),
        PromoShare=("PromoShare", "mean"),
        Sweets=("MntSweetProducts", "median"),
        Education=("Education", lambda s: s.mode()[0]),
        Status=("Marital_Status", lambda s: s.mode()[0]),
    ).reset_index()
    profiles["Name"] = name_segments(profiles, df)
    df["Segment name"] = df["Segment"].map(dict(zip(profiles["Segment"], profiles["Name"])))
    return df, profiles


def describe(p) -> str:
    return (f"Median age {p.Age:.0f}, annual household income ${p.Income:,.0f}, "
            f"{p.Children:.1f} child(ren) on average, most common marital status: {p.Status}, "
            f"most common education level: {p.Education}, annual spending ${p.Spending:,.0f}, "
            f"{p.WebShare:.0%} of purchases made online, {p.PromoShare:.0%} of purchases made "
            f"on promotion, spending on sweet products ${p.Sweets:.0f}/year. "
            f"This segment represents {p.Size} customers.")


# ---------- LLM (Mistral AI) ----------
def call_mistral(api_key: str, messages: list, temperature: float = 0.7,
                 json_mode: bool = True, retries: int = 5) -> str:
    payload = {"model": MISTRAL_MODEL, "temperature": temperature, "messages": messages}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    response = None
    for attempt in range(retries):
        response = requests.post(MISTRAL_URL, headers={"Authorization": f"Bearer {api_key}"},
                                 json=payload, timeout=90)
        if response.status_code == 429:  # too many requests: wait, then retry
            time.sleep(2 * (attempt + 1))
            continue
        response.raise_for_status()
        time.sleep(1.5)  # pause to respect the free-tier rate limit
        return response.json()["choices"][0]["message"]["content"]
    if response is not None:
        response.raise_for_status()
    raise RuntimeError("Mistral did not respond. Please try again in a minute.")


def ask_mistral(api_key: str, system: str, user: str, temperature: float = 0.7) -> dict:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for _ in range(3):
        try:
            result = json.loads(call_mistral(api_key, messages, temperature))
            if isinstance(result, dict):
                return result
        except (ValueError, KeyError):  # invalid JSON: ask again
            continue
    raise RuntimeError("Mistral did not return a usable answer. Please try again.")


# ---------- Verification (Hugging Face) ----------
@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)


def stars(clf, text: str) -> int:
    return int(clf(text, truncation=True)[0]["label"][0])  # "4 stars" -> 4


# ---------- Cleaning the LLM answers ----------
def to_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(to_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(to_text(v) for v in value)
    return "" if value is None else str(value)


def to_list(value) -> list:
    if isinstance(value, list):
        return [to_text(v) for v in value]
    if value:
        return [to_text(value)]
    return []


def to_score(value) -> int:
    digits = "".join(c for c in str(value).split("/")[0] if c.isdigit())
    return max(0, min(int(digits or 0), 10))


def to_ab(value) -> str:
    return "B" if "B" in to_text(value).upper() else "A"


def unwrap(rep: dict, key: str) -> dict:
    # Small models sometimes nest the answer, e.g. {"persona": {...}}
    if key not in rep and len(rep) == 1 and isinstance(next(iter(rep.values())), dict):
        return next(iter(rep.values()))
    return rep


def clean_persona(rep: dict, mode: str) -> dict:
    rep = unwrap(rep, "first_name")
    rep["first_name"] = to_text(rep.get("first_name")) or "?"
    rep["portrait"] = to_text(rep.get("portrait"))
    if mode == "single":
        rep["review"] = to_text(rep.get("review")) or "No review."
        rep["max_price"] = to_text(rep.get("max_price")) or "?"
        rep["likes"] = to_list(rep.get("likes"))
        rep["concerns"] = to_list(rep.get("concerns"))
        rep["purchase_intent"] = to_score(rep.get("purchase_intent"))
    else:
        rep["review_a"] = to_text(rep.get("review_a")) or "No review."
        rep["review_b"] = to_text(rep.get("review_b")) or "No review."
        rep["intent_a"] = to_score(rep.get("intent_a"))
        rep["intent_b"] = to_score(rep.get("intent_b"))
        rep["preference"] = to_ab(rep.get("preference"))
        rep["reason"] = to_text(rep.get("reason"))
    return rep


def clean_summary(s: dict, mode: str) -> dict:
    s = unwrap(s, "verdict")
    texts = (["verdict", "priority_segment", "recommended_price", "slogan"] if mode == "single"
             else ["verdict", "why", "segments_a", "segments_b", "slogan"])
    lists = ["strengths", "risks", "improvements"] if mode == "single" else ["improvements"]
    for key in texts:
        s[key] = to_text(s.get(key))
    for key in lists:
        s[key] = to_list(s.get(key))
    if mode == "ab":
        s["winner"] = to_ab(s.get("winner"))
    return s


# ---------- Panel ----------
def run_panel(api_key: str, profiles: pd.DataFrame, mode: str, products: dict):
    clf = load_sentiment()
    results = []
    bar = st.progress(0.0, text="The panel is gathering...")
    if mode == "single":
        product_txt = f"Product tested: {products['A']}\nPlanned price: €{products['price_A']:.2f}"
        system = SYSTEM_PERSONA
    else:
        product_txt = (f"Version A: {products['A']} (price: €{products['price_A']:.2f})\n"
                       f"Version B: {products['B']} (price: €{products['price_B']:.2f})")
        system = SYSTEM_PERSONA_AB

    for i, p in enumerate(profiles.itertuples()):
        rep = clean_persona(ask_mistral(
            api_key, system,
            f"Your segment: {p.Name}\nProfile of your segment: {describe(p)}\n\n{product_txt}"), mode)
        rep.update(segment=int(p.Segment), segment_name=p.Name, size=int(p.Size),
                   profile=describe(p))
        if mode == "single":
            rep["hf_stars"] = stars(clf, rep["review"])
        else:
            rep["hf_stars_a"] = stars(clf, rep["review_a"])
            rep["hf_stars_b"] = stars(clf, rep["review_b"])
        results.append(rep)
        bar.progress((i + 1) / len(profiles), text=f"{rep['first_name']} has given their opinion")

    bar.progress(1.0, text="Writing the summary...")
    summary_input = [{k: v for k, v in r.items() if k != "profile"} for r in results]
    summary = clean_summary(ask_mistral(
        api_key, SYSTEM_SUMMARY if mode == "single" else SYSTEM_SUMMARY_AB,
        f"{product_txt}\n\nPanel reactions:\n"
        + json.dumps(summary_input, ensure_ascii=False, indent=2),
        temperature=0.3), mode)
    bar.empty()
    return results, summary, product_txt


# ---------- Interface (Streamlit) ----------
st.set_page_config(page_title="FocusFlash", page_icon="⚡", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""
<div class="ff-hero">
  <div class="ff-logo">⚡ FocusFlash</div>
  <div class="ff-tagline">A virtual consumer focus group in 60 seconds</div>
  <div class="ff-steps">
    <div class="ff-step"><b>1</b>Real customer data</div>
    <div class="ff-step"><b>2</b>Consumer segments</div>
    <div class="ff-step"><b>3</b>AI persona panel</div>
    <div class="ff-step"><b>4</b>Summary and recommendations</div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Mistral API key", type="password",
                            value=os.getenv("MISTRAL_API_KEY", ""))
    k = st.slider("Number of segments", 3, 6, 5)
    st.markdown("---")
    st.markdown("**Building blocks**")
    st.caption("📊 Data: Kaggle\n\n🧩 Segmentation: scikit-learn\n\n"
               "🗣️ Personas: Mistral AI\n\n✅ Verification: Hugging Face")

df, profiles = segment(load_data(), k)
color_map = {row.Name: COLORS[row.Segment % len(COLORS)] for row in profiles.itertuples()}
tab1, tab2, tab3, tab4 = st.tabs(["👥 1. Segments", "🗣️ 2. Panel",
                                  "📈 3. Summary", "🎤 4. Interview a persona"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Customers analysed", f"{len(df):,}", "Kaggle dataset"),
                unsafe_allow_html=True)
    c2.markdown(kpi("Segments", str(k), "K-Means, scikit-learn"), unsafe_allow_html=True)
    c3.markdown(kpi("Median income", f"${df['Income'].median():,.0f}",
                    "per household per year"), unsafe_allow_html=True)
    st.write("")
    fig = px.scatter(df, x="Income", y="Spending", color="Segment name", opacity=0.65,
                     color_discrete_map=color_map,
                     labels={"Income": "Annual income ($)", "Spending": "Annual spending ($)",
                             "Segment name": "Segment"})
    fig.update_layout(legend=dict(orientation="h", y=-0.25), margin=dict(t=20),
                      plot_bgcolor="#FAFAFC")
    st.plotly_chart(fig)
    table = profiles.drop(columns="Segment").rename(columns={
        "Name": "Segment", "Size": "Customers", "Age": "Median age", "Income": "Income ($)",
        "Children": "Children (avg)", "Spending": "Spending ($)", "WebShare": "Online purchases",
        "PromoShare": "Promo purchases", "Sweets": "Sweets ($/yr)", "Status": "Marital status"})
    table = table[["Segment", "Customers", "Median age", "Income ($)", "Children (avg)",
                   "Spending ($)", "Online purchases", "Promo purchases", "Sweets ($/yr)",
                   "Education", "Marital status"]]
    st.dataframe(table.style.format({
        "Median age": "{:.0f}", "Income ($)": "{:,.0f}", "Children (avg)": "{:.1f}",
        "Spending ($)": "{:,.0f}", "Online purchases": "{:.0%}", "Promo purchases": "{:.0%}",
        "Sweets ($/yr)": "{:.0f}"}), hide_index=True)

with tab2:
    ab = st.toggle("🆚 Compare two versions (A/B test)")
    products = {}
    if ab:
        ca, cb = st.columns(2)
        with ca:
            products["A"] = st.text_area("Version A", height=110, value=(
                "A plant-based chocolate mousse with no added sugar, sold in packs of two "
                "100 g pots in the supermarket chilled aisle."))
            products["price_A"] = st.number_input("Price of version A (€)", min_value=0.0,
                                                  value=3.50, step=0.10)
        with cb:
            products["B"] = st.text_area("Version B", height=110, value=(
                "A premium plant-based chocolate mousse made with 70% fair-trade dark chocolate, "
                "in a returnable 150 g glass jar."))
            products["price_B"] = st.number_input("Price of version B (€)", min_value=0.0,
                                                  value=4.90, step=0.10)
    else:
        products["A"] = st.text_area("Describe your product", height=100, value=(
            "A plant-based chocolate mousse with no added sugar, sold in packs of two "
            "100 g pots in the supermarket chilled aisle."))
        products["price_A"] = st.number_input("Planned price (€)", min_value=0.0,
                                              value=3.50, step=0.10)

    if st.button("🚀 Run the panel", type="primary"):
        if not api_key:
            st.error("Please add your Mistral API key in the sidebar.")
            st.stop()
        mode = "ab" if ab else "single"
        try:
            results, summary, product_txt = run_panel(api_key, profiles, mode, products)
        except (requests.RequestException, RuntimeError) as e:
            st.error(f"Mistral API error: {e}")
            st.stop()
        st.session_state.update(results=results, summary=summary, mode=mode,
                                product_txt=product_txt, chats={})
        st.success("The panel has given its verdict! See the summary in tab 3.")

    if "results" in st.session_state:
        cols = st.columns(3)
        for i, r in enumerate(st.session_state["results"]):
            cols[i % 3].markdown(persona_card(r, st.session_state["mode"]), unsafe_allow_html=True)

with tab3:
    if "summary" not in st.session_state:
        st.info("Run the panel in tab 2 first.")
    else:
        res = pd.DataFrame(st.session_state["results"])
        s = st.session_state["summary"]
        res["Persona"] = res["first_name"] + " · " + res["segment_name"]
        weights = res["size"] / res["size"].sum()

        st.markdown(f'<div class="ff-verdict">🧭 <b>Verdict:</b> {esc(s["verdict"])}</div>',
                    unsafe_allow_html=True)

        if st.session_state["mode"] == "single":
            intent = (res["purchase_intent"] * weights).sum()
            c1, c2, c3 = st.columns(3)
            c1.markdown(kpi("Weighted purchase intent", f"{intent:.1f}/10",
                            "weighted by segment size", gauge_color(intent)),
                        unsafe_allow_html=True)
            c2.markdown(kpi("Average sentiment (Hugging Face)", f"{res['hf_stars'].mean():.1f}/5",
                            "independent check of the reviews"), unsafe_allow_html=True)
            c3.markdown(kpi("Personas interviewed", str(len(res)),
                            f"representing {res['size'].sum()} customers"), unsafe_allow_html=True)
            st.write("")
            st.markdown(f"**💶 Recommended price:** {s['recommended_price']}")
            st.markdown(f"**🎯 Priority target:** {s['priority_segment']}")

            res["LLM intent (/10)"] = res["purchase_intent"]
            res["HF sentiment (/10)"] = res["hf_stars"] * 2
            fig = px.bar(res, x="Persona", y=["LLM intent (/10)", "HF sentiment (/10)"],
                         barmode="group", title="Consistency check: LLM vs Hugging Face",
                         color_discrete_sequence=["#5B3DF5", "#F59E0B"])
            fig.update_layout(legend_title_text="", xaxis_title="", yaxis_title="Score /10",
                              yaxis_range=[0, 10], plot_bgcolor="#FAFAFC")
            st.plotly_chart(fig)

            a, b, c = st.columns(3)
            a.markdown(box("💪 Strengths", s["strengths"], "#E7F8F1"), unsafe_allow_html=True)
            b.markdown(box("⚠️ Risks", s["risks"], "#FDECEC"), unsafe_allow_html=True)
            c.markdown(box("🛠️ Recommendations", s["improvements"], "#F1EEFF"),
                       unsafe_allow_html=True)
        else:
            int_a = (res["intent_a"] * weights).sum()
            int_b = (res["intent_b"] * weights).sum()
            share_a = weights[res["preference"] == "A"].sum()
            winner = s["winner"]
            c1, c2, c3 = st.columns(3)
            c1.markdown(kpi("Intent for version A", f"{int_a:.1f}/10",
                            "weighted by segment size", "#5B3DF5"), unsafe_allow_html=True)
            c2.markdown(kpi("Intent for version B", f"{int_b:.1f}/10",
                            "weighted by segment size", "#EC4899"), unsafe_allow_html=True)
            c3.markdown(kpi("Customers preferring A", f"{share_a:.0%}",
                            f"vs {1 - share_a:.0%} for B"), unsafe_allow_html=True)
            st.write("")
            st.markdown(f"**🏆 Winning version: {winner}** · {s['why']}")
            st.markdown(f"**🟣 Prefer A:** {s['segments_a']}")
            st.markdown(f"**🩷 Prefer B:** {s['segments_b']}")

            res["Version A"] = res["intent_a"]
            res["Version B"] = res["intent_b"]
            fig = px.bar(res, x="Persona", y=["Version A", "Version B"], barmode="group",
                         title="Purchase intent by persona",
                         color_discrete_sequence=["#5B3DF5", "#EC4899"])
            fig.update_layout(legend_title_text="", xaxis_title="", yaxis_title="Score /10",
                              yaxis_range=[0, 10], plot_bgcolor="#FAFAFC")
            st.plotly_chart(fig)
            st.caption(f"Hugging Face check: average sentiment {res['hf_stars_a'].mean():.1f}/5 "
                       f"for A and {res['hf_stars_b'].mean():.1f}/5 for B.")
            st.markdown(box(f"🛠️ Recommendations for version {winner}", s["improvements"],
                            "#F1EEFF"), unsafe_allow_html=True)

        st.markdown(f'<div class="ff-slogan">💬 “{esc(s["slogan"])}”</div>',
                    unsafe_allow_html=True)
        st.write("")
        st.warning("The personas are simulations: FocusFlash is designed to pre-test an idea "
                   "quickly, not to replace a study with real consumers.")

with tab4:
    if "results" not in st.session_state:
        st.info("Run the panel in tab 2 first.")
    else:
        results = st.session_state["results"]
        idx = st.selectbox("Choose a participant", range(len(results)),
                           format_func=lambda i: f"{results[i]['first_name']} · "
                                                 f"{results[i]['segment_name']}")
        r = results[idx]
        chat = st.session_state["chats"].setdefault(idx, [])
        review = (r["review"] if st.session_state["mode"] == "single"
                  else f"Version A: {r['review_a']} Version B: {r['review_b']} "
                       f"I prefer version {r['preference']}: {r['reason']}")
        system = SYSTEM_INTERVIEW.format(first_name=r["first_name"], portrait=r["portrait"],
                                         segment=r["segment_name"], profile=r["profile"],
                                         product=st.session_state["product_txt"], review=review)

        st.markdown(persona_card(r, st.session_state["mode"]), unsafe_allow_html=True)
        for msg in chat:
            with st.chat_message(msg["role"], avatar="🎤" if msg["role"] == "user" else "🙂"):
                st.write(msg["content"])

        question = None
        qcols = st.columns(len(QUICK_QUESTIONS))
        for qc, q in zip(qcols, QUICK_QUESTIONS):
            if qc.button(q, key=f"q_{idx}_{q}"):
                question = q
        with st.form(f"interview_{idx}", clear_on_submit=True):
            typed = st.text_input(f"Your question for {r['first_name']}")
            if st.form_submit_button("Send") and typed.strip():
                question = typed.strip()

        if question:
            if not api_key:
                st.error("Please add your Mistral API key in the sidebar.")
                st.stop()
            messages = [{"role": "system", "content": system}] + chat + [
                {"role": "user", "content": question}]
            try:
                with st.spinner(f"{r['first_name']} is thinking..."):
                    answer = call_mistral(api_key, messages, temperature=0.7, json_mode=False)
            except (requests.RequestException, RuntimeError) as e:
                st.error(f"Mistral API error: {e}")
                st.stop()
            chat += [{"role": "user", "content": question},
                     {"role": "assistant", "content": answer.strip()}]
            st.rerun()