from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "02-processed"
RESULTS = ROOT / "results"
OUTFILE = RESULTS / "plots.html"

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
QUARTER_LABELS = ["Q1", "Q2", "Q3", "Q4"]

COLORS = {
    "athletes": "#2f6f73",
    "us": "#d88c46",
    "expected": "#8c6bb1",
    "games_2022": "#4f7cac",
    "games_2026": "#c44e52",
    "positive": "#2f6f73",
    "negative": "#c75d5d",
    "gray": "#687076",
    "light_gray": "#d8d2c4",
}

STATE_ABBREVIATIONS = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    athletes = pd.read_csv(PROCESSED / "team_usa_winter_olympians.csv")
    us_monthly = pd.read_csv(PROCESSED / "us_births_monthly_distribution.csv")
    us_quarterly = pd.read_csv(PROCESSED / "us_births_quarterly_distribution.csv")

    athletes = add_athlete_display_columns(athletes)
    us_monthly = add_us_monthly_display_columns(us_monthly)
    us_quarterly = add_us_quarterly_display_columns(us_quarterly)

    return athletes, us_monthly, us_quarterly


def add_athlete_display_columns(athletes: pd.DataFrame) -> pd.DataFrame:
    athletes = athletes.copy()
    athletes["month_label"] = pd.Categorical(
        athletes["birth_month"].map(lambda month: MONTH_LABELS[int(month) - 1]),
        categories=MONTH_LABELS,
        ordered=True,
    )
    athletes["quarter_label"] = pd.Categorical(
        athletes["birth_quarter"].map(lambda quarter: f"Q{int(quarter)}"),
        categories=QUARTER_LABELS,
        ordered=True,
    )
    athletes["medal_status"] = athletes["any_medal"].map({True: "Won medal", False: "No medal"})
    return athletes


def add_us_monthly_display_columns(us_monthly: pd.DataFrame) -> pd.DataFrame:
    us_monthly = us_monthly.copy()
    us_monthly["month_label"] = pd.Categorical(us_monthly["month_name"].str[:3], categories=MONTH_LABELS, ordered=True)
    return us_monthly


def add_us_quarterly_display_columns(us_quarterly: pd.DataFrame) -> pd.DataFrame:
    us_quarterly = us_quarterly.copy()
    us_quarterly["quarter_label"] = pd.Categorical(us_quarterly["quarter_label"], categories=QUARTER_LABELS, ordered=True)
    return us_quarterly


def athlete_month_distribution(athletes: pd.DataFrame, us_monthly: pd.DataFrame) -> pd.DataFrame:
    if "month_label" not in athletes.columns:
        athletes = add_athlete_display_columns(athletes)
    counts = (
        athletes.groupby(["birth_month", "month_label"], observed=True)
        .size()
        .rename("athlete_count")
        .reset_index()
        .sort_values("birth_month")
    )
    counts["athlete_percent"] = counts["athlete_count"] / counts["athlete_count"].sum() * 100

    out = counts.merge(us_monthly[["month", "percent_births"]], left_on="birth_month", right_on="month", how="left")
    out["expected_count"] = out["percent_births"] / 100 * len(athletes)
    out["count_difference"] = out["athlete_count"] - out["expected_count"]
    out["expected_ratio"] = out["athlete_count"] / out["expected_count"]
    return out


def athlete_quarter_distribution(athletes: pd.DataFrame, us_quarterly: pd.DataFrame) -> pd.DataFrame:
    if "quarter_label" not in athletes.columns:
        athletes = add_athlete_display_columns(athletes)
    counts = (
        athletes.groupby(["birth_quarter", "quarter_label"], observed=True)
        .size()
        .rename("athlete_count")
        .reset_index()
        .sort_values("birth_quarter")
    )
    counts["athlete_percent"] = counts["athlete_count"] / counts["athlete_count"].sum() * 100

    out = counts.merge(
        us_quarterly[["birth_quarter", "percent_births"]],
        on="birth_quarter",
        how="left",
    )
    out["expected_count"] = out["percent_births"] / 100 * len(athletes)
    out["count_difference"] = out["athlete_count"] - out["expected_count"]
    out["expected_ratio"] = out["athlete_count"] / out["expected_count"]
    return out


def by_games_quarter_distribution(athletes: pd.DataFrame) -> pd.DataFrame:
    if "quarter_label" not in athletes.columns:
        athletes = add_athlete_display_columns(athletes)
    out = (
        athletes.groupby(["games_year", "birth_quarter", "quarter_label"], observed=True)
        .size()
        .rename("athlete_count")
        .reset_index()
        .sort_values(["games_year", "birth_quarter"])
    )
    out["athlete_percent"] = out["athlete_count"] / out.groupby("games_year")["athlete_count"].transform("sum") * 100
    return out


def sport_distribution(athletes: pd.DataFrame) -> pd.DataFrame:
    sports = athletes.copy()
    sports["sports"] = sports["sports"].fillna("Missing sport")
    out = sports["sports"].value_counts().rename_axis("sport").reset_index(name="athlete_count")
    return out.sort_values("athlete_count", ascending=True)


def medal_quarter_distribution(athletes: pd.DataFrame) -> pd.DataFrame:
    if "quarter_label" not in athletes.columns or "medal_status" not in athletes.columns:
        athletes = add_athlete_display_columns(athletes)
    out = (
        athletes.groupby(["quarter_label", "medal_status"], observed=True)
        .size()
        .rename("athlete_count")
        .reset_index()
    )
    totals = out.groupby("quarter_label", observed=True)["athlete_count"].transform("sum")
    out["percent_within_quarter"] = out["athlete_count"] / totals * 100
    return out


def athlete_month_by_games(athletes: pd.DataFrame) -> pd.DataFrame:
    if "month_label" not in athletes.columns:
        athletes = add_athlete_display_columns(athletes)
    out = (
        athletes.groupby(["games_year", "birth_month", "month_label"], observed=True)
        .size()
        .rename("athlete_count")
        .reset_index()
        .sort_values(["games_year", "birth_month"])
    )
    out["athlete_percent"] = out["athlete_count"] / out.groupby("games_year")["athlete_count"].transform("sum") * 100
    return out


def sport_quarter_distribution(athletes: pd.DataFrame) -> pd.DataFrame:
    if "quarter_label" not in athletes.columns:
        athletes = add_athlete_display_columns(athletes)
    out = (
        athletes.groupby(["sports", "birth_quarter", "quarter_label"], observed=True)
        .size()
        .rename("athlete_count")
        .reset_index()
        .sort_values(["sports", "birth_quarter"])
    )
    out["percent_within_sport"] = out["athlete_count"] / out.groupby("sports")["athlete_count"].transform("sum") * 100
    return out


def parse_birth_state(birthplace: object) -> str | None:
    if not isinstance(birthplace, str) or "(USA)" not in birthplace:
        return None
    place = re.sub(r"\s*\(USA\)\s*$", "", birthplace).strip()
    parts = [part.strip() for part in place.split(",")]
    if len(parts) < 2:
        return None
    return STATE_ABBREVIATIONS.get(parts[-1])


def birth_state_distribution(athletes: pd.DataFrame) -> pd.DataFrame:
    out = athletes.copy()
    out["birth_state"] = out["birthplace"].map(parse_birth_state)
    return (
        out.dropna(subset=["birth_state"])
        .groupby("birth_state")
        .size()
        .rename("athlete_count")
        .reset_index()
        .sort_values("athlete_count", ascending=False)
    )


def density_curve(values: pd.Series, points: int = 140) -> tuple[list[float], list[float]]:
    clean = [float(v) for v in values.dropna()]
    if len(clean) < 2:
        return clean, [0.0 for _ in clean]
    low = min(clean)
    high = max(clean)
    if math.isclose(low, high):
        return [low], [1.0]
    mean = sum(clean) / len(clean)
    variance = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    std = math.sqrt(variance) or 1.0
    bandwidth = 1.06 * std * (len(clean) ** -0.2)
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.0
    xs = [low + (high - low) * i / (points - 1) for i in range(points)]
    coef = 1 / (len(clean) * bandwidth * math.sqrt(2 * math.pi))
    ys = []
    for x in xs:
        density = sum(math.exp(-0.5 * ((x - v) / bandwidth) ** 2) for v in clean) * coef
        ys.append(density)
    return xs, ys


def linear_fit(x_values: pd.Series, y_values: pd.Series) -> tuple[list[float], list[float], float]:
    frame = pd.DataFrame({"x": x_values, "y": y_values}).dropna()
    if len(frame) < 2:
        return [], [], float("nan")
    x_mean = frame["x"].mean()
    y_mean = frame["y"].mean()
    denom = ((frame["x"] - x_mean) ** 2).sum()
    slope = 0.0 if math.isclose(denom, 0.0) else ((frame["x"] - x_mean) * (frame["y"] - y_mean)).sum() / denom
    intercept = y_mean - slope * x_mean
    x_line = [float(frame["x"].min()), float(frame["x"].max())]
    y_line = [intercept + slope * x for x in x_line]
    return x_line, y_line, float(slope)


def fig_month_percent(month_dist: pd.DataFrame) -> dict:
    return {
        "title": "Birth Month Distribution: Team USA Winter Olympians vs U.S. Births",
        "description": "Compares the athlete birth-month share with the baseline U.S. birth distribution from 1995-2005.",
        "data": [
            {
                "type": "bar",
                "name": "Team USA athletes",
                "x": month_dist["month_label"].astype(str).tolist(),
                "y": month_dist["athlete_percent"].round(2).tolist(),
                "marker": {"color": COLORS["athletes"]},
                "hovertemplate": "%{x}<br>Athletes: %{y:.2f}%<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "U.S. births baseline",
                "x": month_dist["month_label"].astype(str).tolist(),
                "y": month_dist["percent_births"].round(2).tolist(),
                "line": {"color": COLORS["us"], "width": 3},
                "marker": {"size": 9},
                "hovertemplate": "%{x}<br>U.S. births: %{y:.2f}%<extra></extra>",
            },
        ],
        "layout": {
            "yaxis": {"title": "Percent of group", "ticksuffix": "%"},
            "xaxis": {"title": "Birth month"},
            "barmode": "group",
        },
    }


def fig_quarter_percent(quarter_dist: pd.DataFrame) -> dict:
    return {
        "title": "Birth Quarter Distribution",
        "description": "Quarter-level view of the relative age effect question. The original hypothesis expects Q1 to be overrepresented.",
        "data": [
            {
                "type": "bar",
                "name": "Team USA athletes",
                "x": quarter_dist["quarter_label"].astype(str).tolist(),
                "y": quarter_dist["athlete_percent"].round(2).tolist(),
                "marker": {"color": COLORS["athletes"]},
                "hovertemplate": "%{x}<br>Athletes: %{y:.2f}%<extra></extra>",
            },
            {
                "type": "bar",
                "name": "U.S. births baseline",
                "x": quarter_dist["quarter_label"].astype(str).tolist(),
                "y": quarter_dist["percent_births"].round(2).tolist(),
                "marker": {"color": COLORS["us"]},
                "hovertemplate": "%{x}<br>U.S. births: %{y:.2f}%<extra></extra>",
            },
        ],
        "layout": {
            "yaxis": {"title": "Percent of group", "ticksuffix": "%"},
            "xaxis": {"title": "Birth quarter"},
            "barmode": "group",
        },
    }


def fig_observed_minus_expected_month(month_dist: pd.DataFrame) -> dict:
    colors = [COLORS["positive"] if val >= 0 else COLORS["negative"] for val in month_dist["count_difference"]]
    return {
        "title": "Observed Athlete Counts Minus Expected Counts by Month",
        "description": "Expected counts are based on the U.S. monthly birth distribution. Positive bars mean more athletes than expected for that month.",
        "data": [
            {
                "type": "bar",
                "name": "Observed - expected",
                "x": month_dist["month_label"].astype(str).tolist(),
                "y": month_dist["count_difference"].round(1).tolist(),
                "marker": {"color": colors},
                "customdata": month_dist[["athlete_count", "expected_count", "expected_ratio"]].round(2).values.tolist(),
                "hovertemplate": (
                    "%{x}<br>Observed: %{customdata[0]:.0f}"
                    "<br>Expected: %{customdata[1]:.1f}"
                    "<br>Observed / expected: %{customdata[2]:.2f}"
                    "<br>Difference: %{y:.1f}<extra></extra>"
                ),
            }
        ],
        "layout": {
            "yaxis": {"title": "Athlete count difference"},
            "xaxis": {"title": "Birth month"},
            "shapes": [{"type": "line", "x0": -0.5, "x1": 11.5, "y0": 0, "y1": 0, "line": {"color": "#333", "width": 1}}],
        },
    }


def fig_games_year_quarter(games_quarter: pd.DataFrame) -> dict:
    data = []
    for year, frame in games_quarter.groupby("games_year"):
        data.append(
            {
                "type": "bar",
                "name": str(year),
                "x": frame["quarter_label"].astype(str).tolist(),
                "y": frame["athlete_percent"].round(2).tolist(),
                "customdata": frame["athlete_count"].tolist(),
                "hovertemplate": "%{x}<br>%{fullData.name}: %{y:.2f}%<br>Count: %{customdata}<extra></extra>",
            }
        )

    return {
        "title": "Birth Quarter Distribution by Games Year",
        "description": "Checks whether the same birth-quarter pattern appears in both Olympic rosters or is driven by one year.",
        "data": data,
        "layout": {
            "yaxis": {"title": "Percent within Games year", "ticksuffix": "%"},
            "xaxis": {"title": "Birth quarter"},
            "barmode": "group",
            "colorway": [COLORS["athletes"], COLORS["us"]],
        },
    }


def fig_age_distribution(athletes: pd.DataFrame) -> dict:
    data = []
    for year, frame in athletes.groupby("games_year"):
        data.append(
            {
                "type": "histogram",
                "name": str(year),
                "x": frame["age_at_games"].round(2).tolist(),
                "opacity": 0.72,
                "xbins": {"start": 14, "end": 56, "size": 2},
                "hovertemplate": "Age: %{x}<br>Count: %{y}<extra></extra>",
            }
        )

    return {
        "title": "Athlete Age at Opening Ceremony",
        "description": "Describes roster age composition. This helps flag whether one Games year has a very different athlete mix.",
        "data": data,
        "layout": {
            "xaxis": {"title": "Age at Games"},
            "yaxis": {"title": "Athlete count"},
            "barmode": "overlay",
            "colorway": [COLORS["games_2022"], COLORS["games_2026"]],
        },
    }


def fig_sport_distribution(sports: pd.DataFrame) -> dict:
    return {
        "title": "Athlete Count by Sport",
        "description": "Shows the composition of the athlete data after filling missing team-event sports from the roster-page supplement. A large sport can heavily influence the overall birth-month pattern.",
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "name": "Athletes",
                "x": sports["athlete_count"].tolist(),
                "y": sports["sport"].tolist(),
                "marker": {"color": COLORS["athletes"]},
                "hovertemplate": "%{y}<br>Athletes: %{x}<extra></extra>",
            }
        ],
        "layout": {
            "xaxis": {"title": "Athlete count"},
            "yaxis": {"title": ""},
            "height": 620,
        },
    }


def fig_medal_by_quarter(medal_quarter: pd.DataFrame) -> dict:
    data = []
    for status in ["Won medal", "No medal"]:
        frame = medal_quarter[medal_quarter["medal_status"] == status]
        data.append(
            {
                "type": "bar",
                "name": status,
                "x": frame["quarter_label"].astype(str).tolist(),
                "y": frame["athlete_count"].tolist(),
                "customdata": frame["percent_within_quarter"].round(1).tolist(),
                "hovertemplate": "%{x}<br>%{fullData.name}<br>Count: %{y}<br>Within-quarter share: %{customdata:.1f}%<extra></extra>",
            }
        )

    return {
        "title": "Medal Outcomes by Birth Quarter",
        "description": "Explores whether medal-winning athlete appearances cluster by birth quarter. This is descriptive only because medal outcomes are sparse.",
        "data": data,
        "layout": {
            "xaxis": {"title": "Birth quarter"},
            "yaxis": {"title": "Athlete count"},
            "barmode": "stack",
            "colorway": [COLORS["expected"], "#d8d2c4"],
        },
    }


def fig_month_count(month_dist: pd.DataFrame) -> dict:
    return {
        "title": "Athlete Count by Birth Month",
        "description": "Raw count plot of athlete-Games appearances by birth month. This shows the same distribution as the percent plot without normalizing to the U.S. baseline.",
        "data": [
            {
                "type": "bar",
                "name": "Athletes",
                "x": month_dist["month_label"].astype(str).tolist(),
                "y": month_dist["athlete_count"].tolist(),
                "marker": {"color": COLORS["athletes"]},
                "hovertemplate": "%{x}<br>Athletes: %{y}<extra></extra>",
            }
        ],
        "layout": {
            "xaxis": {"title": "Birth month"},
            "yaxis": {"title": "Athlete count"},
        },
    }


def fig_month_percent_lines(month_dist: pd.DataFrame) -> dict:
    return {
        "title": "Monthly Percent Lines: Athletes vs U.S. Births",
        "description": "Line-plot version of the core comparison. It makes the direction of the month-to-month differences easy to scan.",
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Team USA athletes",
                "x": month_dist["month_label"].astype(str).tolist(),
                "y": month_dist["athlete_percent"].round(2).tolist(),
                "line": {"color": COLORS["athletes"], "width": 3},
                "hovertemplate": "%{x}<br>Athletes: %{y:.2f}%<extra></extra>",
            },
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "U.S. births baseline",
                "x": month_dist["month_label"].astype(str).tolist(),
                "y": month_dist["percent_births"].round(2).tolist(),
                "line": {"color": COLORS["us"], "width": 3},
                "hovertemplate": "%{x}<br>U.S. births: %{y:.2f}%<extra></extra>",
            },
        ],
        "layout": {
            "xaxis": {"title": "Birth month"},
            "yaxis": {"title": "Percent of group", "ticksuffix": "%"},
        },
    }


def fig_age_density(athletes: pd.DataFrame) -> dict:
    data = []
    for year, frame in athletes.groupby("games_year"):
        xs, ys = density_curve(frame["age_at_games"])
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": str(year),
                "x": [round(x, 2) for x in xs],
                "y": [round(y, 4) for y in ys],
                "fill": "tozeroy",
                "opacity": 0.55,
                "hovertemplate": "Age: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>",
            }
        )
    return {
        "title": "Age Density at Opening Ceremony",
        "description": "Smoothed density view of athlete age. This complements the histogram by making the age-shape comparison between Games years clearer.",
        "data": data,
        "layout": {
            "xaxis": {"title": "Age at Games"},
            "yaxis": {"title": "Estimated density"},
            "colorway": [COLORS["games_2022"], COLORS["games_2026"]],
        },
    }


def fig_age_box_by_quarter(athletes: pd.DataFrame) -> dict:
    if "quarter_label" not in athletes.columns:
        athletes = add_athlete_display_columns(athletes)
    data = []
    for quarter in QUARTER_LABELS:
        frame = athletes[athletes["quarter_label"].astype(str) == quarter]
        data.append(
            {
                "type": "box",
                "name": quarter,
                "y": frame["age_at_games"].round(2).tolist(),
                "marker": {"color": COLORS["athletes"]},
                "boxmean": True,
                "hovertemplate": f"{quarter}<br>Age: %{{y:.2f}}<extra></extra>",
            }
        )
    return {
        "title": "Age Outliers by Birth Quarter",
        "description": "Box plot of age by birth quarter. This checks whether unusual ages are concentrated in one birth-quarter group.",
        "data": data,
        "layout": {
            "xaxis": {"title": "Birth quarter"},
            "yaxis": {"title": "Age at Games"},
        },
    }


def fig_age_violin_by_quarter(athletes: pd.DataFrame) -> dict:
    if "quarter_label" not in athletes.columns:
        athletes = add_athlete_display_columns(athletes)
    data = []
    for quarter in QUARTER_LABELS:
        frame = athletes[athletes["quarter_label"].astype(str) == quarter]
        data.append(
            {
                "type": "violin",
                "name": quarter,
                "y": frame["age_at_games"].round(2).tolist(),
                "box": {"visible": True},
                "meanline": {"visible": True},
                "points": False,
                "hovertemplate": f"{quarter}<br>Age: %{{y:.2f}}<extra></extra>",
            }
        )
    return {
        "title": "Age Distribution Shape by Birth Quarter",
        "description": "Violin plot showing the full age distribution inside each birth quarter, including spread and central tendency.",
        "data": data,
        "layout": {
            "xaxis": {"title": "Birth quarter"},
            "yaxis": {"title": "Age at Games"},
            "colorway": ["#4f7cac", "#2f6f73", "#8c6bb1", "#c44e52"],
        },
    }


def fig_age_strip_by_quarter(athletes: pd.DataFrame) -> dict:
    if "quarter_label" not in athletes.columns:
        athletes = add_athlete_display_columns(athletes)
    frame = athletes.sort_values(["birth_quarter", "age_at_games", "name"]).reset_index(drop=True)
    jitter = [((i % 9) - 4) * 0.035 for i in range(len(frame))]
    x = [float(q) + j for q, j in zip(frame["birth_quarter"], jitter)]
    return {
        "title": "Individual Athlete Ages by Birth Quarter",
        "description": "Strip plot of individual athlete-Games rows. Each point is one athlete appearance, which helps reveal outliers hidden by aggregate summaries.",
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "name": "Athlete-Games",
                "x": x,
                "y": frame["age_at_games"].round(2).tolist(),
                "customdata": frame[["name", "games_year", "month_label", "sports"]].astype(str).values.tolist(),
                "marker": {"color": COLORS["athletes"], "opacity": 0.68, "size": 8},
                "hovertemplate": (
                    "%{customdata[0]} (%{customdata[1]})"
                    "<br>Birth month: %{customdata[2]}"
                    "<br>Sport: %{customdata[3]}"
                    "<br>Age: %{y:.2f}<extra></extra>"
                ),
            }
        ],
        "layout": {
            "xaxis": {"title": "Birth quarter", "tickmode": "array", "tickvals": [1, 2, 3, 4], "ticktext": QUARTER_LABELS},
            "yaxis": {"title": "Age at Games"},
        },
    }


def fig_age_birth_month_scatter(athletes: pd.DataFrame) -> dict:
    if "month_label" not in athletes.columns:
        athletes = add_athlete_display_columns(athletes)
    data = []
    for year, frame in athletes.groupby("games_year"):
        frame = frame.sort_values(["birth_month", "age_at_games", "name"]).reset_index(drop=True)
        jitter = [((i % 7) - 3) * 0.04 for i in range(len(frame))]
        data.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": str(year),
                "x": [float(month) + j for month, j in zip(frame["birth_month"], jitter)],
                "y": frame["age_at_games"].round(2).tolist(),
                "customdata": frame[["name", "month_label", "sports"]].astype(str).values.tolist(),
                "marker": {"size": 8, "opacity": 0.72},
                "hovertemplate": (
                    "%{customdata[0]}<br>Birth month: %{customdata[1]}"
                    "<br>Sport: %{customdata[2]}"
                    "<br>Age: %{y:.2f}<extra></extra>"
                ),
            }
        )
    return {
        "title": "Age vs Birth Month",
        "description": "Scatter plot checking whether the birth-month pattern is tangled with athlete age or Games year.",
        "data": data,
        "layout": {
            "xaxis": {"title": "Birth month", "tickmode": "array", "tickvals": list(range(1, 13)), "ticktext": MONTH_LABELS},
            "yaxis": {"title": "Age at Games"},
            "colorway": [COLORS["games_2022"], COLORS["games_2026"]],
        },
    }


def fig_age_birth_month_regression(athletes: pd.DataFrame) -> dict:
    if "month_label" not in athletes.columns:
        athletes = add_athlete_display_columns(athletes)
    x_line, y_line, slope = linear_fit(athletes["birth_month"], athletes["age_at_games"])
    return {
        "title": "Regression Check: Age vs Birth Month",
        "description": f"Regression-style diagnostic for whether athlete age changes systematically across birth months. The fitted slope is {slope:.3f} age-years per month.",
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "name": "Athlete-Games",
                "x": athletes["birth_month"].tolist(),
                "y": athletes["age_at_games"].round(2).tolist(),
                "customdata": athletes[["name", "games_year", "month_label", "sports"]].astype(str).values.tolist(),
                "marker": {"color": COLORS["gray"], "opacity": 0.45, "size": 7},
                "hovertemplate": (
                    "%{customdata[0]} (%{customdata[1]})"
                    "<br>Birth month: %{customdata[2]}"
                    "<br>Sport: %{customdata[3]}"
                    "<br>Age: %{y:.2f}<extra></extra>"
                ),
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Linear fit",
                "x": x_line,
                "y": [round(y, 2) for y in y_line],
                "line": {"color": COLORS["negative"], "width": 3},
                "hovertemplate": "Birth month: %{x:.1f}<br>Fitted age: %{y:.2f}<extra></extra>",
            },
        ],
        "layout": {
            "xaxis": {"title": "Birth month", "tickmode": "array", "tickvals": list(range(1, 13)), "ticktext": MONTH_LABELS},
            "yaxis": {"title": "Age at Games"},
        },
    }


def fig_sport_quarter_heatmap(sport_quarter: pd.DataFrame) -> dict:
    pivot = (
        sport_quarter.pivot(index="sports", columns="quarter_label", values="athlete_count")
        .reindex(columns=QUARTER_LABELS)
        .fillna(0)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]
    return {
        "title": "Sport by Birth Quarter Heatmap",
        "description": "Heatmap of athlete counts by sport and birth quarter. This checks whether the overall quarter pattern is driven by particular sports.",
        "data": [
            {
                "type": "heatmap",
                "x": pivot.columns.astype(str).tolist(),
                "y": pivot.index.tolist(),
                "z": pivot.values.tolist(),
                "colorscale": [[0, "#f4f8f7"], [1, COLORS["athletes"]]],
                "hovertemplate": "%{y}<br>%{x}: %{z} athletes<extra></extra>",
            }
        ],
        "layout": {
            "xaxis": {"title": "Birth quarter"},
            "yaxis": {"title": "Sport"},
            "height": 660,
        },
    }


def fig_games_quarter_point(games_quarter: pd.DataFrame) -> dict:
    data = []
    for year, frame in games_quarter.groupby("games_year"):
        data.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": str(year),
                "x": frame["quarter_label"].astype(str).tolist(),
                "y": frame["athlete_percent"].round(2).tolist(),
                "customdata": frame["athlete_count"].tolist(),
                "marker": {"size": 11},
                "hovertemplate": "%{fullData.name}<br>%{x}: %{y:.2f}%<br>Count: %{customdata}<extra></extra>",
            }
        )
    return {
        "title": "Point Plot: Birth Quarter by Games Year",
        "description": "Point-plot version of the Games-year comparison. Parallel lines suggest the quarter pattern is similar in both rosters.",
        "data": data,
        "layout": {
            "xaxis": {"title": "Birth quarter"},
            "yaxis": {"title": "Percent within Games year", "ticksuffix": "%"},
            "colorway": [COLORS["games_2022"], COLORS["games_2026"]],
        },
    }


def fig_sport_pie(sports: pd.DataFrame) -> dict:
    top = sports.sort_values("athlete_count", ascending=False).head(8).copy()
    other_count = sports["athlete_count"].sum() - top["athlete_count"].sum()
    if other_count > 0:
        top = pd.concat([top, pd.DataFrame([{"sport": "Other sports", "athlete_count": other_count}])], ignore_index=True)
    return {
        "title": "Sport Composition of Athlete-Games Rows",
        "description": "Pie chart showing roster composition by sport. This is context for interpreting whether one large sport, especially ice hockey, may affect overall birth-month patterns.",
        "data": [
            {
                "type": "pie",
                "labels": top["sport"].tolist(),
                "values": top["athlete_count"].tolist(),
                "hole": 0.38,
                "sort": False,
                "hovertemplate": "%{label}<br>Athletes: %{value}<br>Share: %{percent}<extra></extra>",
            }
        ],
        "layout": {
            "height": 560,
            "showlegend": True,
        },
    }


def fig_numeric_pair_plot(athletes: pd.DataFrame) -> dict:
    if "quarter_label" not in athletes.columns:
        athletes = add_athlete_display_columns(athletes)
    dimensions = [
        {"label": "Birth month", "values": athletes["birth_month"].tolist()},
        {"label": "Birth year", "values": athletes["birth_year"].tolist()},
        {"label": "Age at Games", "values": athletes["age_at_games"].round(2).tolist()},
        {"label": "Events entered", "values": athletes["events_entered"].tolist()},
        {"label": "Medal count", "values": athletes["medal_count"].tolist()},
    ]
    return {
        "title": "Pair Plot of Numeric Athlete Variables",
        "description": "Scatter-matrix view of numeric variables. This is a broad relationship scan for age, birth timing, event count, and medal count.",
        "data": [
            {
                "type": "splom",
                "dimensions": dimensions,
                "marker": {
                    "color": athletes["birth_quarter"].tolist(),
                    "colorscale": [[0, "#4f7cac"], [0.33, "#2f6f73"], [0.66, "#8c6bb1"], [1, "#c44e52"]],
                    "showscale": False,
                    "opacity": 0.55,
                    "size": 6,
                },
                "diagonal": {"visible": False},
                "showupperhalf": False,
            }
        ],
        "layout": {
            "height": 780,
        },
    }


def fig_month_by_games_facets(month_games: pd.DataFrame) -> dict:
    frame_2022 = month_games[month_games["games_year"] == 2022]
    frame_2026 = month_games[month_games["games_year"] == 2026]
    return {
        "title": "Faceted Birth-Month Counts by Games Year",
        "description": "Facet-style bar charts split by Games year. This checks whether the monthly distribution changes between 2022 and 2026.",
        "data": [
            {
                "type": "bar",
                "name": "2022",
                "x": frame_2022["month_label"].astype(str).tolist(),
                "y": frame_2022["athlete_count"].tolist(),
                "marker": {"color": COLORS["games_2022"]},
                "xaxis": "x",
                "yaxis": "y",
                "hovertemplate": "2022<br>%{x}: %{y} athletes<extra></extra>",
            },
            {
                "type": "bar",
                "name": "2026",
                "x": frame_2026["month_label"].astype(str).tolist(),
                "y": frame_2026["athlete_count"].tolist(),
                "marker": {"color": COLORS["games_2026"]},
                "xaxis": "x2",
                "yaxis": "y2",
                "hovertemplate": "2026<br>%{x}: %{y} athletes<extra></extra>",
            },
        ],
        "layout": {
            "xaxis": {"title": "2022 birth month", "domain": [0.0, 0.47]},
            "yaxis": {"title": "Athlete count"},
            "xaxis2": {"title": "2026 birth month", "domain": [0.53, 1.0]},
            "yaxis2": {"title": "Athlete count", "anchor": "x2", "matches": "y"},
            "annotations": [
                {"text": "2022", "x": 0.235, "xref": "paper", "y": 1.08, "yref": "paper", "showarrow": False, "font": {"size": 15}},
                {"text": "2026", "x": 0.765, "xref": "paper", "y": 1.08, "yref": "paper", "showarrow": False, "font": {"size": 15}},
            ],
            "showlegend": False,
        },
    }


def fig_birth_state_choropleth(state_counts: pd.DataFrame) -> dict:
    return {
        "title": "U.S. Birthplace Map for Team USA Winter Olympians",
        "description": "Choropleth of U.S.-born athlete-Games rows by birth state. This gives geographic context because access to winter sports is not evenly distributed.",
        "data": [
            {
                "type": "choropleth",
                "locationmode": "USA-states",
                "locations": state_counts["birth_state"].tolist(),
                "z": state_counts["athlete_count"].tolist(),
                "colorscale": [[0, "#f4f8f7"], [1, COLORS["athletes"]]],
                "colorbar": {"title": "Athletes"},
                "hovertemplate": "%{location}<br>Athlete-Games: %{z}<extra></extra>",
            }
        ],
        "layout": {
            "geo": {"scope": "usa"},
            "height": 580,
        },
    }


def build_figures() -> tuple[list[dict], dict[str, str]]:
    athletes, us_monthly, us_quarterly = read_data()
    month_dist = athlete_month_distribution(athletes, us_monthly)
    quarter_dist = athlete_quarter_distribution(athletes, us_quarterly)
    sports = sport_distribution(athletes)
    medal_quarter = medal_quarter_distribution(athletes)
    sport_quarter = sport_quarter_distribution(athletes)
    state_counts = birth_state_distribution(athletes)

    figures = [
        fig_month_percent(month_dist),
        fig_quarter_percent(quarter_dist),
        fig_observed_minus_expected_month(month_dist),
        fig_age_density(athletes),
        fig_age_box_by_quarter(athletes),
        fig_age_violin_by_quarter(athletes),
        fig_sport_distribution(sports),
        fig_sport_quarter_heatmap(sport_quarter),
        fig_medal_by_quarter(medal_quarter),
        fig_numeric_pair_plot(athletes),
        fig_birth_state_choropleth(state_counts),
    ]

    q1 = quarter_dist.loc[quarter_dist["quarter_label"].astype(str) == "Q1"].iloc[0]
    q2 = quarter_dist.loc[quarter_dist["quarter_label"].astype(str) == "Q2"].iloc[0]
    q3 = quarter_dist.loc[quarter_dist["quarter_label"].astype(str) == "Q3"].iloc[0]
    q4 = quarter_dist.loc[quarter_dist["quarter_label"].astype(str) == "Q4"].iloc[0]
    max_month = month_dist.loc[month_dist["athlete_count"].idxmax()]
    min_month = month_dist.loc[month_dist["athlete_count"].idxmin()]

    summary = {
        "athlete_rows": f"{len(athletes):,}",
        "games_years": ", ".join(str(year) for year in sorted(athletes["games_year"].unique())),
        "q1": f"{q1['athlete_percent']:.1f}% athletes vs {q1['percent_births']:.1f}% U.S. births",
        "q2": f"{q2['athlete_percent']:.1f}% athletes vs {q2['percent_births']:.1f}% U.S. births",
        "q3": f"{q3['athlete_percent']:.1f}% athletes vs {q3['percent_births']:.1f}% U.S. births",
        "q4": f"{q4['athlete_percent']:.1f}% athletes vs {q4['percent_births']:.1f}% U.S. births",
        "max_month": f"{max_month['month_label']} ({int(max_month['athlete_count'])} athletes)",
        "min_month": f"{min_month['month_label']} ({int(min_month['athlete_count'])} athletes)",
    }

    return figures, summary


def render_html(figures: list[dict], summary: dict[str, str]) -> str:
    figure_blocks = []
    for idx, figure in enumerate(figures):
        div_id = f"plot-{idx}"
        spec = {
            "data": figure["data"],
            "layout": {
                "template": "plotly_white",
                "title": {"text": figure["title"], "x": 0.02, "xanchor": "left"},
                "font": {"family": "Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"},
                "margin": {"l": 72, "r": 32, "t": 82, "b": 62},
                "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
                **figure["layout"],
            },
            "config": {"responsive": True, "displaylogo": False},
        }
        figure_blocks.append(
            f"""
            <section class="plot-card">
              <p>{figure["description"]}</p>
              <div id="{div_id}" class="plot"></div>
              <script>
                Plotly.newPlot("{div_id}", {json.dumps(spec["data"])}, {json.dumps(spec["layout"])}, {json.dumps(spec["config"])});
              </script>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>COGS 108 EDA Plots</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2628;
      --muted: #5f686b;
      --line: #d9dedb;
      --paper: #fbfaf7;
      --panel: #ffffff;
      --accent: #2f6f73;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--paper);
    }}
    header {{
      padding: 36px max(24px, calc((100vw - 1120px) / 2)) 20px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: clamp(28px, 4vw, 44px);
      line-height: 1.05;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    p {{
      color: var(--muted);
      line-height: 1.55;
      margin: 0;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      margin-top: 22px;
    }}
    .metric {{
      border-left: 4px solid var(--accent);
      background: #f4f8f7;
      padding: 12px 14px;
      min-height: 86px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
      margin-bottom: 6px;
    }}
    .metric strong {{
      display: block;
      font-size: 20px;
      line-height: 1.2;
    }}
    .plot-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px 18px 8px;
      margin-bottom: 20px;
    }}
    .plot-card p {{
      max-width: 850px;
      margin-bottom: 8px;
    }}
    .plot {{
      width: 100%;
      min-height: 460px;
    }}
    @media (max-width: 700px) {{
      header {{ padding: 28px 18px 18px; }}
      main {{ padding: 18px; }}
      .plot-card {{ padding: 14px 10px 4px; }}
      .plot {{ min-height: 420px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Birth Month EDA for Team USA Winter Olympians</h1>
    <p>
      Interactive plots for exploring whether the 2022 and 2026 U.S. Winter Olympic rosters have birth-month
      patterns that differ from U.S. births from 1995-2005.
    </p>
    <div class="summary">
      <div class="metric"><span>Athlete-Games Rows</span><strong>{summary["athlete_rows"]}</strong></div>
      <div class="metric"><span>Games Years</span><strong>{summary["games_years"]}</strong></div>
      <div class="metric"><span>Q1</span><strong>{summary["q1"]}</strong></div>
      <div class="metric"><span>Q2</span><strong>{summary["q2"]}</strong></div>
      <div class="metric"><span>Q3</span><strong>{summary["q3"]}</strong></div>
      <div class="metric"><span>Q4</span><strong>{summary["q4"]}</strong></div>
      <div class="metric"><span>Most Common Month</span><strong>{summary["max_month"]}</strong></div>
      <div class="metric"><span>Least Common Month</span><strong>{summary["min_month"]}</strong></div>
    </div>
  </header>
  <main>
    {"".join(figure_blocks)}
  </main>
</body>
</html>
"""


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    figures, summary = build_figures()
    OUTFILE.write_text(render_html(figures, summary), encoding="utf-8")
    print(f"Wrote {OUTFILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
