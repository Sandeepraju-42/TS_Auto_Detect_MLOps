"""
Synthetic fashion SKU demand generator.

Produces a panel at grain (sku_id, date), with a deliberate, controllable 
data-generating process (DGP) rather than pure noise. basically, every regressor 
has a real, known effect on demand, so later EDA should actually recover these 
relationships, and later model comparisons have a genuine ground truth to be 
graded against.

Two outputs at the end:
  - `full` dataframe: everything, including true_demand and pattern_type.
    These are generator ground truth ONLY -- true_demand is what
    sales_qty would have been without stockout censoring, and pattern_type
    is which trend/seasonal/cyclic recipe drove a SKU. Use these to grade
    EDA findings and model bias, never as model inputs.
  - `model_ready` dataframe: the full frame with true_demand and
    pattern_type dropped, and endogenous feedback columns (review_count,
    avg_rating, wishlist_adds, social_trend_score) still in *current-period*
    form -- feature_engineering.py's add_endogenous_lagged_features() is
    what lags/drops those, not this script. This matches your stated
    assumption that the production pipeline has no oracle type labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from dependancies.params_utils import load_params

# --------------------------------------------------------------------------
# Config -- from params.yaml's data_generation section, not hardcoded, so
# `dvc repro` reruns this (and everything downstream) whenever you change
# seed/n_skus/n_weeks there instead of silently going stale.
# --------------------------------------------------------------------------
_p = load_params()["data_generation"]
SEED = _p["seed"]
N_SKUS = _p["n_skus"]
START_DATE = _p["start_date"]
N_WEEKS = _p["n_weeks"]     # ~5 years at the default: enough history for
                             # seasonal models plus VALIDATION_HORIZON=104 +
                             # FORECAST_HORIZON=8 with room to spare

CATEGORIES = ["Tops", "Bottoms", "Dresses", "Outerwear", "Footwear", "Accessories"]
SUB_CATEGORY = {
    "Tops": ["T-Shirt", "Blouse", "Shirt", "Sweater"],
    "Bottoms": ["Jeans", "Trousers", "Shorts", "Skirt"],
    "Dresses": ["Casual Dress", "Evening Dress", "Sundress"],
    "Outerwear": ["Jacket", "Coat", "Parka"],
    "Footwear": ["Sneakers", "Boots", "Sandals", "Heels"],
    "Accessories": ["Bag", "Belt", "Scarf", "Hat"],
}
COLORS = ["Black", "White", "Beige", "Navy", "Red", "Pastel Pink", "Olive", "Grey"]
MATERIALS = ["Cotton", "Denim", "Wool", "Synthetic", "Leather", "Linen"]
BRANDS = ["ZARA", "H&M", "COS", "Mango", "Uniqlo"]
GENDERS = ["Male", "Female", "Unisex"]
SIZE_RANGES = ["XS-S", "S-M", "M-L", "L-XL"]

# category -> (pattern_type, seasonal_peak_week, seasonal_amplitude, weather_sensitivity)
# pattern_type mirrors the TYPE_LABEL_MAP taxonomy from the forecast-framework
# project (trend / seasonal / trend_seasonal / cyclic) so this data can slot
# straight into that pipeline's per-category algorithm pools.
CATEGORY_PROFILE = {
    "Tops":        dict(pattern_type="seasonal",      peak_week=26, amplitude=0.35, weather_sign=+1),
    "Bottoms":     dict(pattern_type="trend",          peak_week=20, amplitude=0.10, weather_sign=0),
    "Dresses":     dict(pattern_type="trend_seasonal", peak_week=24, amplitude=0.50, weather_sign=+1),
    "Outerwear":   dict(pattern_type="seasonal",       peak_week=50, amplitude=0.60, weather_sign=-1),
    "Footwear":    dict(pattern_type="trend",          peak_week=30, amplitude=0.15, weather_sign=0),
    "Accessories": dict(pattern_type="cyclic",         peak_week=None, amplitude=0.20, weather_sign=0),
}
PRICE_ELASTICITY = {  # demand multiplier = (price / base_price) ** -elasticity
    "Tops": 1.2, "Bottoms": 1.0, "Dresses": 1.4,
    "Outerwear": 0.8, "Footwear": 1.1, "Accessories": 0.9,
}

rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------
# 1. SKU master (static attributes)
# --------------------------------------------------------------------------
def make_sku_master(n_skus: int) -> pd.DataFrame:
    categories = rng.choice(CATEGORIES, size=n_skus)
    rows = []
    for i, cat in enumerate(categories):
        sub_cat = rng.choice(SUB_CATEGORY[cat])
        base_price = rng.uniform(20, 150) if cat != "Outerwear" else rng.uniform(60, 250)

        # 80% of SKUs launch at the very start of history (gives most SKUs a
        # full, stable series); 20% launch later, staggered across the
        # timeline, so lifecycle features (new/growth/mature/decline) have
        # real examples to learn from instead of being constant.
        if rng.random() < 0.8:
            launch_week = 0
        else:
            launch_week = int(rng.integers(1, N_WEEKS - 60))

        rows.append(dict(
            sku_id=f"SKU{i:05d}",
            category=cat,
            sub_category=sub_cat,
            color=rng.choice(COLORS),
            material=rng.choice(MATERIALS),
            brand=rng.choice(BRANDS),
            gender=rng.choice(GENDERS),
            size_range=rng.choice(SIZE_RANGES),
            base_price=round(base_price, 2),
            launch_week=launch_week,
            lead_time_days=int(rng.choice([14, 21, 30, 45])),
            store_count=int(rng.integers(20, 400)),
            channel=rng.choice(["online", "in-store", "both"], p=[0.3, 0.3, 0.4]),
            # base demand level: lognormal so a few SKUs are hits and most
            # are modest -- matches the skewed target distribution EDA
            # step 3 (step3_target_distribution) is written to expect.
            base_demand_level=rng.lognormal(mean=3.0, sigma=0.6),
            quality_score=float(np.clip(rng.normal(4.0, 0.4), 2.5, 5.0)),  # drives avg_rating later
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Calendar (deterministic from date, shared across all SKUs)
# --------------------------------------------------------------------------
def make_calendar(start_date: str, n_weeks: int) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")
    df = pd.DataFrame({"date": dates})
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = False  # week-start rows only, N/A at weekly grain

    # simplified fixed holiday weeks (ISO week numbers based on EU holiday season)
    holiday_weeks = {1, 7, 20, 47, 48, 51, 52}
    df["is_holiday"] = df["week_of_year"].isin(holiday_weeks)
    
    # assunimg last 3 weeks of Feb/May/Sep/Nov fashion weeks
    df["fashion_week_flag"] = df["week_of_year"].isin({7, 8, 9, 20, 21, 22, 36, 37, 38, 47, 48, 49})  

    # assuming 3 weeks of August     
    df["back_to_school_flag"] = df["week_of_year"].isin({33, 34, 35})
    
    # first week of month
    df["payday_week_flag"] = df["date"].dt.day <= 7                   
    return df



# --------------------------------------------------------------------------
# 3. Weather (seasonal sinusoid + noise, Northern-hemisphere shaped)
# --------------------------------------------------------------------------
def make_weather(calendar: pd.DataFrame) -> pd.DataFrame:
    df = calendar[["date", "week_of_year"]].copy()
    # temperature peaks ~week 30 (late July), troughs ~week 4 (late Jan)
    seasonal = 15 * np.sin(2 * np.pi * (df["week_of_year"] - 30) / 52)
    df["temperature"] = 12 + seasonal + rng.normal(0, 2.5, len(df))
    df["precipitation"] = np.clip(
        rng.gamma(shape=2.0, scale=8.0, size=len(df)) + 3 * np.sin(2 * np.pi * df["week_of_year"] / 52),
        0, None,
    )
    return df[["date", "temperature", "precipitation"]]


# --------------------------------------------------------------------------
# 4. Price / promo / marketing panel (per SKU, per week)
# --------------------------------------------------------------------------
def make_price_promo_marketing(sku_master: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    panel = sku_master[["sku_id", "base_price"]].merge(calendar[["date"]], how="cross")
    panel = panel.sort_values(["sku_id", "date"]).reset_index(drop=True)
    n = len(panel)

    # promo_flag: independent Bernoulli bursts rather than iid per-row noise
    # -- real promos run for a few consecutive weeks, not single isolated
    # weeks, so this is generated per-SKU as short "on" runs.
    promo_flag = np.zeros(n, dtype=bool)
    for sku, idx in panel.groupby("sku_id").indices.items():
        idx = np.array(idx)
        t = 0
        while t < len(idx):
            if rng.random() < 0.06:  # ~6% weekly chance a promo run starts
                run_len = int(rng.integers(1, 4))
                promo_flag[idx[t: t + run_len]] = True
                t += run_len
            else:
                t += 1
    panel["promo_flag"] = promo_flag

    panel["discount_pct"] = np.where(
        panel["promo_flag"], rng.uniform(0.1, 0.5, n), 0.0
    )
    panel["markdown_stage"] = pd.cut(
        panel["discount_pct"], bins=[-0.01, 0, 0.15, 0.3, 1.0],
        labels=["none", "light", "moderate", "deep"],
    )
    panel["price"] = (panel["base_price"] * (1 - panel["discount_pct"])).round(2)
    panel["competitor_price_index"] = rng.normal(1.0, 0.05, n).clip(0.8, 1.2)

    # marketing spend correlates with promo (campaigns usually back promos)
    # plus an independent baseline so it's not a pure duplicate of promo_flag.
    panel["marketing_spend"] = (
        rng.gamma(2.0, 150, n) + panel["promo_flag"] * rng.uniform(500, 2000, n)
    ).round(2)
    panel["campaign_flag"] = panel["promo_flag"] & (rng.random(n) < 0.7)

    return panel


# --------------------------------------------------------------------------
# 5. Cross-product: pair each SKU with an in-category substitute
# --------------------------------------------------------------------------
def add_cross_product_features(panel: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    substitute_map = {}
    for cat, group in sku_master.groupby("category"):
        ids = group["sku_id"].tolist()
        shuffled = rng.permutation(ids)
        # pair sku[i] with sku[i+1] (wrapping) as its "substitute" --
        # simple, deterministic, and guarantees every SKU has exactly one.
        for i, sku in enumerate(ids):
            substitute_map[sku] = shuffled[(list(shuffled).index(sku) + 1) % len(shuffled)]

    sub_price = panel[["sku_id", "date", "price"]].rename(
        columns={"sku_id": "substitute_id", "price": "substitute_price"}
    )
    panel = panel.copy()
    panel["substitute_id"] = panel["sku_id"].map(substitute_map)
    panel = panel.merge(sub_price, on=["substitute_id", "date"], how="left")
    panel["substitute_price_ratio"] = (panel["price"] / panel["substitute_price"]).round(3)
    # cannibalization_index: how much cheaper the substitute is, floored at 0
    # -- only a cheaper substitute should pull demand away, not a pricier one.
    panel["cannibalization_index"] = (1 - panel["substitute_price_ratio"]).clip(lower=0)
    return panel.drop(columns=["substitute_id", "substitute_price"])


# --------------------------------------------------------------------------
# 6. Demand DGP -- combine every regressor into true_demand
# --------------------------------------------------------------------------
def simulate_true_demand(panel: pd.DataFrame, sku_master: pd.DataFrame, calendar: pd.DataFrame,
                          weather: pd.DataFrame) -> pd.DataFrame:
    df = panel.merge(sku_master, on="sku_id", suffixes=("", "_master"))
    df = df.merge(calendar, on="date").merge(weather, on="date")
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    profile = df["category"].map(CATEGORY_PROFILE)
    df["pattern_type"] = profile.apply(lambda p: p["pattern_type"])
    amplitude = profile.apply(lambda p: p["amplitude"])
    peak_week = profile.apply(lambda p: p["peak_week"] if p["peak_week"] is not None else 26)
    weather_sign = profile.apply(lambda p: p["weather_sign"])
    elasticity = df["category"].map(PRICE_ELASTICITY)

    # week index since series start, used for trend and long cyclic components
    week_idx = (df["date"] - df["date"].min()).dt.days // 7

    # -- trend component: only "trend" and "trend_seasonal" categories drift
    is_trend_like = df["pattern_type"].isin(["trend", "trend_seasonal"])
    trend = 1 + is_trend_like * 0.0015 * week_idx  # slow ~0.15%/week compounding growth

    # -- seasonal component: annual sinusoid keyed to each category's peak week
    seasonal = 1 + amplitude * np.sin(2 * np.pi * (df["week_of_year"] - peak_week + 13) / 52)
    seasonal = np.where(df["pattern_type"].isin(["seasonal", "trend_seasonal"]), seasonal, 1.0)

    # -- cyclic component: ~2-year business-cycle-like wave, unrelated to the
    # calendar -- this is what should make "cyclic" category models fail if
    # you only give them calendar/seasonal features, by design.
    cyclic = 1 + amplitude * np.sin(2 * np.pi * week_idx / 104)
    cyclic = np.where(df["pattern_type"] == "cyclic", cyclic, 1.0)

    # -- price elasticity effect
    price_effect = (df["price"] / df["base_price"]) ** (-elasticity)

    # -- promo uplift, on top of the price effect itself
    promo_effect = np.where(df["promo_flag"], rng.uniform(1.3, 2.0, len(df)), 1.0)

    # -- weather effect: deviation from that week's seasonal norm, signed per
    # category (outerwear up when colder than normal, tops up when warmer)
    seasonal_norm_temp = 12 + 15 * np.sin(2 * np.pi * (df["week_of_year"] - 30) / 52)
    temp_deviation = (df["temperature"] - seasonal_norm_temp) / 10
    weather_effect = 1 + weather_sign * 0.15 * temp_deviation

    # -- marketing: diminishing-returns (log) uplift
    marketing_effect = 1 + 0.08 * np.log1p(df["marketing_spend"] / 500)

    # -- cannibalization: cheaper substitute pulls demand down
    cannibal_effect = 1 - 0.3 * df["cannibalization_index"]

    # -- lifecycle: ramp up over first 8 weeks post-launch, slow decline
    # after week 150 of a SKU's own life (mirrors real product lifecycles)
    weeks_since_launch = week_idx - df["launch_week"]
    ramp = np.clip(weeks_since_launch / 8, 0, 1)
    decline = np.where(weeks_since_launch > 150, 1 - 0.002 * (weeks_since_launch - 150), 1.0)
    lifecycle_effect = np.clip(ramp * decline, 0, None)

    noise = rng.lognormal(mean=0, sigma=0.25, size=len(df))

    true_demand = (
        df["base_demand_level"] * trend * seasonal * cyclic * price_effect
        * promo_effect * weather_effect * marketing_effect * cannibal_effect
        * lifecycle_effect * noise
    )
    # rows before a SKU's own launch shouldn't exist at all -- drop them
    # instead of zeroing, since a true zero is meaningfully different from
    # "not on sale yet".
    df["true_demand"] = np.round(true_demand).clip(lower=0)
    df["days_since_launch"] = weeks_since_launch * 7
    df = df[weeks_since_launch >= 0].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# 7. Inventory simulation -> stockout censoring -> sales_qty
# --------------------------------------------------------------------------
def simulate_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sequential per-SKU process (can't vectorize across time since inventory
    at t depends on sales at t-1) -- this is what actually creates the
    over/under-forecasting story: a replenishment policy that doesn't
    perfectly track demand produces realistic, structural censoring rather
    than random missingness.
    """
    df = df.sort_values(["sku_id", "date"]).copy()
    sales_qty = np.zeros(len(df))
    inventory_on_hand = np.zeros(len(df))
    stockout_flag = np.zeros(len(df), dtype=bool)

    for sku, idx in df.groupby("sku_id").indices.items():
        idx = np.array(idx)
        demand = df["true_demand"].values[idx]
        lead_weeks = max(1, int(df["lead_time_days"].values[idx[0]] / 7))

        # replenish to a target of ~4 weeks of trailing average demand,
        # but only every `lead_weeks` periods, and only ordering ~90% of
        # the target on average -- a deliberately imperfect policy, since a
        # perfect one would never produce interesting stockouts to detect.
        target_cover_weeks = 4
        stock = demand[:lead_weeks].mean() * target_cover_weeks if len(demand) else 0
        for t in range(len(idx)):
            if t > 0 and t % lead_weeks == 0:
                trailing = demand[max(0, t - 8):t].mean() if t > 0 else demand[0]
                order_qty = trailing * target_cover_weeks * rng.uniform(0.75, 1.05)
                stock += order_qty

            sold = min(demand[t], stock)
            sales_qty[idx[t]] = sold
            inventory_on_hand[idx[t]] = stock
            stockout_flag[idx[t]] = demand[t] > stock
            stock = max(0, stock - sold)

    df["sales_qty"] = np.round(sales_qty)
    df["inventory_on_hand"] = np.round(inventory_on_hand)
    df["stockout_flag"] = stockout_flag
    return df


# --------------------------------------------------------------------------
# 8. Endogenous feedback (reviews, ratings, wishlist, trend score)
# --------------------------------------------------------------------------
def simulate_endogenous_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """
    These are generated FROM sales_qty (this period's), which is exactly
    why feature_engineering.py's add_endogenous_lagged_features() must lag
    them by one period before any model sees them -- using the same-period
    values here would be feeding the model something downstream of the
    target it's trying to predict.
    """
    df = df.sort_values(["sku_id", "date"]).copy()
    df["review_count"] = np.round(df["sales_qty"] * rng.uniform(0.05, 0.15, len(df))).clip(0)
    df["avg_rating"] = (df["quality_score"] + rng.normal(0, 0.15, len(df))).clip(1, 5).round(1)
    df["wishlist_adds"] = np.round(df["sales_qty"] * rng.uniform(0.2, 0.6, len(df))
                                    + rng.poisson(3, len(df))).clip(0)
    # social_trend_score: a smoothed, noisy proxy for "search interest",
    # loosely trailing demand rather than a pure copy of it.
    trend_base = df.groupby("sku_id")["sales_qty"].transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    df["social_trend_score"] = (trend_base / (trend_base.max() + 1e-6) * 100
                                 + rng.normal(0, 5, len(df))).clip(0, 100).round(1)
    return df


# --------------------------------------------------------------------------
# 9. Assemble
# --------------------------------------------------------------------------
def build_dataset():
    sku_master = make_sku_master(N_SKUS)
    calendar = make_calendar(START_DATE, N_WEEKS)
    weather = make_weather(calendar)

    panel = make_price_promo_marketing(sku_master, calendar)
    panel = add_cross_product_features(panel, sku_master)

    df = simulate_true_demand(panel, sku_master, calendar, weather)
    df = simulate_inventory(df)
    df = simulate_endogenous_feedback(df)

    full_cols = [
        "sku_id", "date", "category", "sub_category", "color", "material", "brand",
        "gender", "size_range", "price", "discount_pct", "promo_flag", "markdown_stage",
        "competitor_price_index", "week_of_year", "month", "quarter", "is_holiday",
        "is_weekend", "fashion_week_flag", "back_to_school_flag", "payday_week_flag",
        "days_since_launch", "temperature", "precipitation", "marketing_spend",
        "campaign_flag", "lead_time_days", "store_count", "channel",
        "review_count", "avg_rating", "wishlist_adds", "social_trend_score",
        "inventory_on_hand", "stockout_flag", "cannibalization_index",
        "substitute_price_ratio", "pattern_type", "true_demand", "sales_qty",
    ]
    full = df[full_cols].reset_index(drop=True)

    model_ready = full.drop(columns=["pattern_type", "true_demand"])
    return full, model_ready


if __name__ == "__main__":
    full, model_ready = build_dataset()

    project_dir = Path(__file__).resolve().parent
    input_dir = project_dir / "input"
    input_dir.mkdir(exist_ok=True, parents=True)

    full_path = input_dir / "synthetic_fashion_demand_full.csv"
    model_ready_path = input_dir / "synthetic_fashion_demand.csv"

    full.to_csv(full_path, index=False)
    model_ready.to_csv(model_ready_path, index=False)

    print(f"Saved full dataset to: {full_path}")
    print(f"Saved model-ready dataset to: {model_ready_path}")
    print(f"full shape: {full.shape}")
    print(f"model_ready shape: {model_ready.shape}")
    print(f"SKUs: {full['sku_id'].nunique()}, date range: {full['date'].min()} to {full['date'].max()}")
    print(f"overall stockout rate: {full['stockout_flag'].mean():.2%}")
    print(f"mean (true_demand - sales_qty) on stockout rows: "
          f"{(full.loc[full.stockout_flag, 'true_demand'] - full.loc[full.stockout_flag, 'sales_qty']).mean():.2f}")