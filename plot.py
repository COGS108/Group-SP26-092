from __future__ import annotations

import json
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
    "positive": "#2f6f73",
    "negative": "#c75d5d",
    "gray": "#687076",
}


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    athletes = pd.read_csv(PROCESSED / "team_usa_winter_olympians.csv")
    us_monthly = pd.read_csv(PROCESSED / "us_births_monthly_distribution.csv")
    us_quarterly = pd.read_csv(PROCESSED / "us_births_quarterly_distribution.csv")

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

    us_monthly["month_label"] = pd.Categorical(us_monthly["month_name"].str[:3], categories=MONTH_LABELS, ordered=True)
    us_quarterly["quarter_label"] = pd.Categorical(us_quarterly["quarter_label"], categories=QUARTER_LABELS, ordered=True)

    return athletes, us_monthly, us_quarterly


def athlete_month_distribution(athletes: pd.DataFrame, us_monthly: pd.DataFrame) -> pd.DataFrame:
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
    sports["sports"] = sports["sports"].fillna("Team sport/no individual result")
    out = sports["sports"].value_counts().rename_axis("sport").reset_index(name="athlete_count")
    return out.sort_values("athlete_count", ascending=True)


def medal_quarter_distribution(athletes: pd.DataFrame) -> pd.DataFrame:
    out = (
        athletes.groupby(["quarter_label", "medal_status"], observed=True)
        .size()
        .rename("athlete_count")
        .reset_index()
    )
    totals = out.groupby("quarter_label", observed=True)["athlete_count"].transform("sum")
    out["percent_within_quarter"] = out["athlete_count"] / totals * 100
    return out


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
            "colorway": [COLORS["athletes"], COLORS["us"]],
        },
    }


def fig_sport_distribution(sports: pd.DataFrame) -> dict:
    return {
        "title": "Athlete Count by Sport / Result Category",
        "description": "Shows the composition of the athlete data. A large sport can heavily influence the overall birth-month pattern.",
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


def build_figures() -> tuple[list[dict], dict[str, str]]:
    athletes, us_monthly, us_quarterly = read_data()
    month_dist = athlete_month_distribution(athletes, us_monthly)
    quarter_dist = athlete_quarter_distribution(athletes, us_quarterly)
    games_quarter = by_games_quarter_distribution(athletes)
    sports = sport_distribution(athletes)
    medal_quarter = medal_quarter_distribution(athletes)

    figures = [
        fig_month_percent(month_dist),
        fig_quarter_percent(quarter_dist),
        fig_observed_minus_expected_month(month_dist),
        fig_games_year_quarter(games_quarter),
        fig_age_distribution(athletes),
        fig_sport_distribution(sports),
        fig_medal_by_quarter(medal_quarter),
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
