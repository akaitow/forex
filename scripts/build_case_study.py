from __future__ import annotations

from itertools import combinations
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CURRENCIES = ["EUR", "USD", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD"]
START_DATE = "2021-08-01"
END_DATE = "2026-08-01"
CACHE_PATH = DATA_DIR / "fx_prices_2021-08-01_2026-08-01.csv"
NOTEBOOK_PATH = ROOT / "fx_currency_strength_case_study.ipynb"


def download_prices() -> tuple[pd.DataFrame, dict[str, str]]:
    series_by_pair: dict[str, pd.Series] = {}
    tickers_by_pair: dict[str, str] = {}

    for first, second in combinations(CURRENCIES, 2):
        loaded = False
        for base, quote in ((first, second), (second, first)):
            pair = base + quote
            ticker = f"{pair}=X"
            raw = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if raw.empty or "Close" not in raw:
                continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors="coerce").dropna()
            if close.empty:
                continue
            close.name = pair
            series_by_pair[pair] = close
            tickers_by_pair[pair] = ticker
            loaded = True
            break
        if not loaded:
            raise RuntimeError(f"No Yahoo Finance series found for {first}/{second}")

    prices = pd.concat(series_by_pair.values(), axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.to_csv(CACHE_PATH, index_label="Date")
    return prices, tickers_by_pair


def build_strength(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_returns = np.log(prices / prices.shift(1)).mul(100).dropna(how="any")
    currency_strength = pd.DataFrame(
        0.0, index=pair_returns.index, columns=CURRENCIES
    )
    for pair in pair_returns.columns:
        base, quote = pair[:3], pair[3:]
        currency_strength[base] += pair_returns[pair] / 7
        currency_strength[quote] -= pair_returns[pair] / 7
    return pair_returns, currency_strength


def block_bootstrap_ci(
    values: np.ndarray, *, block_length: int = 10, simulations: int = 5000, seed: int = 42
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    block_starts = np.arange(n - block_length + 1)
    means = np.empty(simulations)
    blocks_needed = int(np.ceil(n / block_length))
    for simulation in range(simulations):
        starts = rng.choice(block_starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[start : start + block_length] for start in starts])[:n]
        means[simulation] = sample.mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def newey_west_regression(x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> dict[str, float]:
    x_matrix = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.solve(x_matrix.T @ x_matrix, x_matrix.T @ y)
    residuals = y - x_matrix @ beta
    meat = np.zeros((2, 2))
    for t in range(len(y)):
        xt = x_matrix[t][:, None]
        meat += residuals[t] ** 2 * (xt @ xt.T)
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        cross = np.zeros((2, 2))
        for t in range(lag, len(y)):
            xt = x_matrix[t][:, None]
            xlag = x_matrix[t - lag][:, None]
            cross += residuals[t] * residuals[t - lag] * (xt @ xlag.T)
        meat += weight * (cross + cross.T)
    bread = np.linalg.inv(x_matrix.T @ x_matrix)
    covariance = bread @ meat @ bread
    standard_error = float(np.sqrt(covariance[1, 1]))
    slope = float(beta[1])
    t_stat = slope / standard_error
    p_value = float(2 * stats.t.sf(abs(t_stat), df=len(y) - 2))
    return {
        "intercept": float(beta[0]),
        "slope": slope,
        "slope_se": standard_error,
        "slope_p": p_value,
    }


def analyse(prices: pd.DataFrame) -> dict[str, object]:
    pair_returns, currency_strength = build_strength(prices)
    available_pairs = set(pair_returns.columns)

    def directed_return(base: str, quote: str) -> pd.Series:
        direct = base + quote
        if direct in available_pairs:
            return pair_returns[direct]
        return -pair_returns[quote + base]

    triangle_residuals = pd.concat(
        [
            directed_return(first, second)
            + directed_return(second, third)
            + directed_return(third, first)
            for first, second, third in combinations(CURRENCIES, 3)
        ],
        axis=1,
    )

    usd_spoke_strength = pd.DataFrame(
        0.0, index=pair_returns.index, columns=CURRENCIES
    )
    for currency in CURRENCIES:
        if currency != "USD":
            usd_spoke_strength[currency] = directed_return(currency, "USD")
    usd_spoke_strength = usd_spoke_strength.sub(
        usd_spoke_strength.mean(axis=1), axis=0
    )

    usd_top4 = usd_spoke_strength.apply(
        lambda row: list(row.nlargest(4).index), axis=1
    )
    usd_rows = []
    for index in range(len(usd_top4) - 1):
        selected = usd_top4.iloc[index]
        usd_rows.append(
            {
                "p0": usd_spoke_strength.iloc[index][selected].mean(),
                "p1": usd_spoke_strength.iloc[index + 1][selected].mean(),
            }
        )
    usd_results = pd.DataFrame(usd_rows)
    usd_regression = newey_west_regression(
        usd_results["p0"].to_numpy(), usd_results["p1"].to_numpy()
    )
    groups = pd.DataFrame(index=currency_strength.index)
    groups["top_4"] = currency_strength.apply(lambda row: list(row.nlargest(4).index), axis=1)
    groups["bottom_4"] = currency_strength.apply(lambda row: list(row.nsmallest(4).index), axis=1)

    rows = []
    for index in range(len(groups) - 1):
        top = groups.iloc[index]["top_4"]
        bottom = groups.iloc[index]["bottom_4"]
        today_strength = currency_strength.iloc[index]
        next_strength = currency_strength.iloc[index + 1]
        rows.append(
            {
                "date": groups.index[index],
                "next_date": groups.index[index + 1],
                "top4": ", ".join(top),
                "p0_top4_avg": today_strength[top].mean(),
                "p1_top4_avg": next_strength[top].mean(),
                "p1_bottom4_avg": next_strength[bottom].mean(),
            }
        )
    results = pd.DataFrame(rows).set_index("date")
    results["continuation"] = results["p1_top4_avg"] > 0

    ci_low, ci_high = block_bootstrap_ci(results["p1_top4_avg"].to_numpy())
    regression = newey_west_regression(
        results["p0_top4_avg"].to_numpy(), results["p1_top4_avg"].to_numpy()
    )
    correlation, correlation_p = stats.pearsonr(
        results["p0_top4_avg"], results["p1_top4_avg"]
    )
    sign_test = stats.binomtest(int(results["continuation"].sum()), len(results), 0.5)

    return {
        "pair_returns": pair_returns,
        "currency_strength": currency_strength,
        "groups": groups,
        "results": results,
        "mean": float(results["p1_top4_avg"].mean()),
        "median": float(results["p1_top4_avg"].median()),
        "std": float(results["p1_top4_avg"].std()),
        "positive_share": float(results["continuation"].mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "correlation": float(correlation),
        "correlation_p": float(correlation_p),
        "sign_p": float(sign_test.pvalue),
        "triangle_median_abs": float(triangle_residuals.abs().stack().median()),
        "triangle_p95_abs": float(triangle_residuals.abs().stack().quantile(0.95)),
        "usd_spoke_mean": float(usd_results["p1"].mean()),
        "usd_spoke_positive_share": float((usd_results["p1"] > 0).mean()),
        "usd_spoke_slope": float(usd_regression["slope"]),
        "usd_spoke_slope_p": float(usd_regression["slope_p"]),
        **regression,
    }


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook(summary: dict[str, object], tickers_by_pair: dict[str, str]) -> None:
    n = len(summary["results"])
    observations = len(summary["pair_returns"])
    start = summary["pair_returns"].index.min().date().isoformat()
    end = summary["pair_returns"].index.max().date().isoformat()
    mean = summary["mean"]
    ci_low = summary["ci_low"]
    ci_high = summary["ci_high"]
    positive_share = summary["positive_share"]
    slope = summary["slope"]
    slope_p = summary["slope_p"]
    correlation = summary["correlation"]
    correlation_p = summary["correlation_p"]
    sign_p = summary["sign_p"]
    triangle_median_abs = summary["triangle_median_abs"]
    triangle_p95_abs = summary["triangle_p95_abs"]
    usd_spoke_slope = summary["usd_spoke_slope"]
    usd_spoke_slope_p = summary["usd_spoke_slope_p"]

    cells = [
        md(
            rf"""
            # Do Today’s Strongest Currencies Stay Strong Tomorrow?

            ### A five-year case study of momentum versus mean reversion across the eight major currencies

            **Decision question:** Does ranking the major currencies by today’s cross-market strength create a useful one-day forecast?

            **Scope:** {start} to {end} · {observations:,} synchronized daily return observations · 28 currency pairs · 8 currencies

            ---

            ## TL;DR

            The four strongest currencies on day $t$ earned an average **{mean:+.4f}%** strength score on day $t+1$. A moving-block bootstrap 95% confidence interval was **[{ci_low:+.4f}%, {ci_high:+.4f}%]**, which includes zero. Continuation occurred on **{positive_share:.1%}** of days.

            The unconditional mean is close to zero, but the dependence tests reveal a small reversal pattern. Continuation occurred on **{positive_share:.1%}** of days (two-sided sign-test $p={sign_p:.3f}$), and the Newey–West $p_0 \rightarrow p_1$ slope was **{slope:+.3f}** ($p={slope_p:.3f}$). The Pearson correlation was **{correlation:+.3f}** ($p={correlation_p:.3f}$).

            **Finding:** the data reject a one-day momentum story and instead show **weak statistical mean reversion**: more extreme day-0 leaders tended to give back a small amount of strength on day 1. The average effect is only about {abs(mean) * 100:.2f} basis points in this non-tradable index, and it fades at longer daily horizons.

            > **Stakeholder implication:** Do not promote yesterday’s top-four ranking as a momentum signal. Treat the reversal pattern as a research lead—not a trading strategy—until it survives a pre-registered out-of-sample test and transaction-cost modeling.
            """
        ),
        md(
            r"""
            ## Context & Methods

            ### Purpose

            Currency pairs make relative moves easy to observe but individual currencies harder to compare. This project creates a single daily strength score for each of the eight major currencies, ranks those scores, and asks whether the leaders retain strength on the next trading day.

            ### Hypothesis

            - **Null ($H_0$):** the top-four group’s expected next-day strength is zero, and today’s group strength ($p_0$) does not predict next-day strength ($p_1$).
            - **Momentum alternative:** $E[p_1] > 0$ and/or the $p_0 \rightarrow p_1$ slope is positive.
            - **Mean-reversion alternative:** $E[p_1] < 0$ and/or the slope is negative.

            ### Analytical flow

            `28 FX closing-price series` → `daily log returns` → `8 currency-strength scores` → `top four on day t` → `same four currencies on day t+1`

            ### Key assumptions

            - Yahoo Finance daily closes are treated as synchronized observations; dates missing from any pair are excluded.
            - Log returns are used because reversing a quote changes only the sign, which preserves the cross-currency accounting identity.
            - Each currency score is the equal-weighted average contribution from its seven pairs.
            - The strength score is an analytical index, not a directly investable portfolio return.
            - The primary test is one day ahead. Confidence intervals use a 10-day moving-block bootstrap; regression inference uses Newey–West standard errors with five lags.
            """
        ),
        code(
            """
            # Setup: imports, parameters, and a restrained visual system
            from itertools import combinations
            from pathlib import Path
            import warnings

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy import stats
            import yfinance as yf
            from IPython.display import display

            warnings.filterwarnings("ignore", category=FutureWarning)
            pd.set_option("display.max_columns", 20)
            pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

            CURRENCIES = ["EUR", "USD", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD"]
            START_DATE = "2021-08-01"
            END_DATE = "2026-08-01"
            CACHE_PATH = Path("data/fx_prices_2021-08-01_2026-08-01.csv")
            RANDOM_SEED = 42

            BLUE = "#2F5D8C"
            BLUE_LIGHT = "#B8CCE0"
            GOLD = "#C6922B"
            INK = "#25313C"
            GREY = "#7A8793"
            GRID = "#DCE2E7"
            BG = "#FAFBFC"

            sns.set_theme(style="whitegrid", context="notebook")
            plt.rcParams.update({
                "figure.facecolor": BG,
                "axes.facecolor": BG,
                "axes.edgecolor": GREY,
                "axes.labelcolor": INK,
                "axes.titlecolor": INK,
                "text.color": INK,
                "xtick.color": INK,
                "ytick.color": INK,
                "grid.color": GRID,
                "grid.linewidth": 0.8,
                "font.family": "DejaVu Sans",
            })
            """
        ),
        md(
            r"""
            ## Data

            The universe contains all $\binom{8}{2}=28$ unique combinations of EUR, USD, JPY, GBP, CHF, AUD, CAD, and NZD. The repository includes a frozen CSV snapshot for reproducibility. If the file is missing—for example in a standalone Colab upload—the next cell downloads the same date range and recreates it.

            **Source:** Yahoo Finance via [`yfinance`](https://ranaroussi.github.io/yfinance/). Data are intended for research and demonstration, not production trading.
            """
        ),
        code(
            """
            def download_fx_prices(currencies, start_date, end_date):
                series_by_pair = {}
                ticker_map = {}

                for first, second in combinations(currencies, 2):
                    for base, quote in ((first, second), (second, first)):
                        pair = base + quote
                        ticker = f"{pair}=X"
                        raw = yf.download(
                            ticker,
                            start=start_date,
                            end=end_date,
                            interval="1d",
                            auto_adjust=False,
                            progress=False,
                            threads=False,
                        )
                        if raw.empty or "Close" not in raw:
                            continue
                        close = raw["Close"]
                        if isinstance(close, pd.DataFrame):
                            close = close.iloc[:, 0]
                        close = pd.to_numeric(close, errors="coerce").dropna()
                        if close.empty:
                            continue
                        close.name = pair
                        series_by_pair[pair] = close
                        ticker_map[pair] = ticker
                        break
                    else:
                        raise RuntimeError(f"No series found for {first}/{second}")

                prices = pd.concat(series_by_pair.values(), axis=1).sort_index()
                prices.index = pd.to_datetime(prices.index).tz_localize(None)
                return prices, ticker_map


            if CACHE_PATH.exists():
                prices = pd.read_csv(CACHE_PATH, index_col="Date", parse_dates=True)
                ticker_map = {column: f"{column}=X" for column in prices.columns}
                data_source = f"Cached snapshot: {CACHE_PATH.as_posix()}"
            else:
                prices, ticker_map = download_fx_prices(CURRENCIES, START_DATE, END_DATE)
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                prices.to_csv(CACHE_PATH, index_label="Date")
                data_source = "Fresh Yahoo Finance download (then cached locally)"

            print(data_source)
            print(f"Raw shape: {prices.shape[0]:,} dates × {prices.shape[1]} pairs")
            display(prices.head(3))
            """
        ),
        code(
            """
            # Data-quality checks at the analytical grain
            expected_pairs = len(list(combinations(CURRENCIES, 2)))
            audit = pd.Series({
                "expected_unique_pairs": expected_pairs,
                "downloaded_pair_series": prices.shape[1],
                "raw_trading_dates": prices.shape[0],
                "dates_complete_across_all_pairs": int(prices.notna().all(axis=1).sum()),
                "duplicate_dates": int(prices.index.duplicated().sum()),
                "nonpositive_prices": int((prices <= 0).sum().sum()),
                "max_pair_missing_share": prices.isna().mean().max(),
            }).to_frame("value")

            assert prices.shape[1] == expected_pairs, "The 28-pair universe is incomplete."
            assert not prices.columns.duplicated().any(), "Duplicate pair columns found."
            assert prices.index.is_monotonic_increasing, "Dates must be sorted."
            assert audit.loc["duplicate_dates", "value"] == 0, "Duplicate dates found."
            assert audit.loc["nonpositive_prices", "value"] == 0, "Prices must be positive."
            display(audit)
            """
        ),
        code(
            """
            # Convert pair returns into equal-weighted individual-currency strength
            pair_returns = np.log(prices / prices.shift(1)).mul(100).dropna(how="any")

            currency_strength = pd.DataFrame(
                0.0, index=pair_returns.index, columns=CURRENCIES
            )
            pair_counts = pd.Series(0, index=CURRENCIES, dtype=int)

            for pair in pair_returns.columns:
                base, quote = pair[:3], pair[3:]
                currency_strength[base] += pair_returns[pair] / 7
                currency_strength[quote] -= pair_returns[pair] / 7
                pair_counts[[base, quote]] += 1

            assert pair_counts.eq(7).all(), "Every currency must contribute through seven pairs."
            assert np.allclose(currency_strength.sum(axis=1), 0, atol=1e-10)

            print(
                f"Analytical sample: {len(currency_strength):,} synchronized daily returns "
                f"from {currency_strength.index.min().date()} to {currency_strength.index.max().date()}"
            )
            display(currency_strength.head(3))
            """
        ),
        code(
            """
            # Coherence check: cross rates should approximately close around every currency triangle
            available_pairs = set(pair_returns.columns)

            def directed_return(base, quote):
                direct = base + quote
                if direct in available_pairs:
                    return pair_returns[direct]
                return -pair_returns[quote + base]


            triangle_residuals = pd.concat([
                directed_return(first, second)
                + directed_return(second, third)
                + directed_return(third, first)
                for first, second, third in combinations(CURRENCIES, 3)
            ], axis=1)

            triangle_audit = pd.Series({
                "currency_triangles": triangle_residuals.shape[1],
                "median_absolute_cycle_residual_pct": triangle_residuals.abs().stack().median(),
                "p95_absolute_cycle_residual_pct": triangle_residuals.abs().stack().quantile(0.95),
            }).to_frame("value")
            display(triangle_audit)
            """
        ),
        md(
            rf"""
            ### Data-source coherence caveat

            Perfectly synchronized FX quotes would sum to zero around every three-currency cycle. The median absolute cycle discrepancy is **{triangle_median_abs:.4f}%**, while the 95th percentile is **{triangle_p95_abs:.4f}%**. The typical mismatch is small, but the upper tail confirms that these public closing series are not an institutional synchronized panel. A seven-USD-leg sensitivity test later checks whether this redundancy drives the main result.
            """
        ),
        md(
            r"""
            ### Why the two four-currency groups mirror each other

            Every pair contributes the same return once positively (to its base currency) and once negatively (to its quote currency). Therefore, on every date:

            $$\sum_{c=1}^{8} \text{strength}_{c,t}=0$$

            Splitting all eight currencies into two equal groups implies:

            $$\operatorname{mean}(\text{top 4})=-\operatorname{mean}(\text{bottom 4})$$

            This is an accounting identity created by the feature engineering—not a market finding. The analysis therefore keeps the top-four signal as the primary outcome and uses the bottom four only to verify the identity.
            """
        ),
        code(
            """
            # Rank currencies on day t, then measure the same names on day t+1
            groups = pd.DataFrame(index=currency_strength.index)
            groups["top_4"] = currency_strength.apply(
                lambda row: list(row.nlargest(4).index), axis=1
            )
            groups["bottom_4"] = currency_strength.apply(
                lambda row: list(row.nsmallest(4).index), axis=1
            )

            records = []
            for position in range(len(groups) - 1):
                top = groups.iloc[position]["top_4"]
                bottom = groups.iloc[position]["bottom_4"]
                today_strength = currency_strength.iloc[position]
                next_strength = currency_strength.iloc[position + 1]
                records.append({
                    "date": groups.index[position],
                    "next_date": groups.index[position + 1],
                    "top4": ", ".join(top),
                    "p0_top4_avg": today_strength[top].mean(),
                    "p1_top4_avg": next_strength[top].mean(),
                    "p1_bottom4_avg": next_strength[bottom].mean(),
                })

            results = pd.DataFrame(records).set_index("date")
            results["continuation"] = results["p1_top4_avg"] > 0
            results["symmetry_error"] = (
                results["p1_top4_avg"] + results["p1_bottom4_avg"]
            )

            assert results["symmetry_error"].abs().max() < 1e-10
            display(results.head())
            """
        ),
        md("## Results"),
        code(
            """
            def block_bootstrap_mean_ci(values, block_length=10, simulations=5_000, seed=42):
                values = np.asarray(values, dtype=float)
                rng = np.random.default_rng(seed)
                n = len(values)
                starts_available = np.arange(n - block_length + 1)
                blocks_needed = int(np.ceil(n / block_length))
                bootstrap_means = np.empty(simulations)
                for simulation in range(simulations):
                    starts = rng.choice(starts_available, size=blocks_needed, replace=True)
                    sample = np.concatenate([
                        values[start:start + block_length] for start in starts
                    ])[:n]
                    bootstrap_means[simulation] = sample.mean()
                return np.quantile(bootstrap_means, [0.025, 0.975])


            next_day = results["p1_top4_avg"].to_numpy()
            ci_low, ci_high = block_bootstrap_mean_ci(next_day, seed=RANDOM_SEED)
            t_test = stats.ttest_1samp(next_day, popmean=0)
            sign_test = stats.binomtest(int((next_day > 0).sum()), len(next_day), p=0.5)

            scorecard = pd.DataFrame({
                "metric": [
                    "Next-day observations",
                    "Mean p1 strength (%)",
                    "Median p1 strength (%)",
                    "Block-bootstrap 95% CI (%)",
                    "Continuation rate",
                    "One-sample t-test p-value",
                    "Two-sided sign-test p-value",
                ],
                "value": [
                    f"{len(next_day):,}",
                    f"{next_day.mean():+.4f}",
                    f"{np.median(next_day):+.4f}",
                    f"[{ci_low:+.4f}, {ci_high:+.4f}]",
                    f"{(next_day > 0).mean():.1%}",
                    f"{t_test.pvalue:.3f}",
                    f"{sign_test.pvalue:.3f}",
                ],
            })
            display(scorecard.style.hide(axis="index"))
            """
        ),
        code(
            """
            # Structural check: top and bottom groups are exact mirrors
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            ax.scatter(
                results["p1_top4_avg"],
                results["p1_bottom4_avg"],
                s=22, alpha=0.45, color=BLUE, edgecolor="none"
            )
            limit = np.abs(results[["p1_top4_avg", "p1_bottom4_avg"]].to_numpy()).max()
            ax.plot([-limit, limit], [limit, -limit], color=GOLD, lw=2, ls="--")
            ax.axhline(0, color=GREY, lw=1)
            ax.axvline(0, color=GREY, lw=1)
            ax.set(
                xlabel="Top-four next-day average strength, p1 (%)",
                ylabel="Bottom-four next-day average strength (%)",
            )
            ax.set_title(
                "Top-four and bottom-four next-day strength",
                loc="left", pad=34, fontsize=14, fontweight="bold"
            )
            ax.text(
                0.00, 1.015,
                "Each point lies on y = −x by construction; this is a validation check, not evidence of predictability.",
                transform=ax.transAxes, color=GREY, fontsize=9.5
            )
            sns.despine(ax=ax)
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            # Primary outcome: the next-day distribution is centered close to zero
            fig, ax = plt.subplots(figsize=(9.5, 5.2))
            sns.histplot(
                results["p1_top4_avg"], bins=36, stat="density",
                color=BLUE_LIGHT, edgecolor=BLUE, linewidth=0.7, ax=ax
            )
            sns.kdeplot(results["p1_top4_avg"], color=BLUE, lw=2, ax=ax)
            ax.axvline(0, color=INK, lw=1.4, ls="--", label="No edge (0%)")
            ax.axvline(results["p1_top4_avg"].mean(), color=GOLD, lw=2,
                       label=f"Observed mean ({results['p1_top4_avg'].mean():+.4f}%)")
            ax.axvspan(ci_low, ci_high, color=GOLD, alpha=0.14, label="95% block-bootstrap CI")
            ax.set(
                xlabel="Top-four next-day average strength, p1 (%)",
                ylabel="Density",
            )
            ax.set_title(
                "Distribution of next-day strength for the prior day’s top four",
                loc="left", pad=34, fontsize=14, fontweight="bold"
            )
            ax.text(
                0.00, 1.015,
                f"n = {len(results):,} trading days · daily log-return strength · {results.index.min().date()} to {results.index.max().date()}",
                transform=ax.transAxes, color=GREY, fontsize=9.5
            )
            ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
            sns.despine(ax=ax)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            rf"""
            The estimated next-day mean is **{mean:+.4f}%**, and the 95% block-bootstrap interval spans zero. That mean test alone does not establish an unconditional edge. However, positive outcomes occur on only **{positive_share:.1%}** of days, and the two-sided sign test ($p={sign_p:.3f}$) points to a small directional tilt toward reversal. The mean and sign results differ because the mean is more sensitive to a few large positive days.
            """
        ),
        code(
            """
            def newey_west_regression(x, y, max_lag=5):
                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
                design = np.column_stack([np.ones(len(x)), x])
                beta = np.linalg.solve(design.T @ design, design.T @ y)
                residuals = y - design @ beta
                meat = np.zeros((2, 2))

                for position in range(len(y)):
                    vector = design[position][:, None]
                    meat += residuals[position] ** 2 * (vector @ vector.T)

                for lag in range(1, max_lag + 1):
                    weight = 1 - lag / (max_lag + 1)
                    cross = np.zeros((2, 2))
                    for position in range(lag, len(y)):
                        current = design[position][:, None]
                        previous = design[position - lag][:, None]
                        cross += (
                            residuals[position] * residuals[position - lag]
                            * (current @ previous.T)
                        )
                    meat += weight * (cross + cross.T)

                bread = np.linalg.inv(design.T @ design)
                covariance = bread @ meat @ bread
                slope_se = np.sqrt(covariance[1, 1])
                slope_t = beta[1] / slope_se
                slope_p = 2 * stats.t.sf(abs(slope_t), df=len(y) - 2)
                return {
                    "intercept": beta[0], "slope": beta[1],
                    "slope_se": slope_se, "slope_p": slope_p,
                }


            x = results["p0_top4_avg"].to_numpy()
            y = results["p1_top4_avg"].to_numpy()
            regression = newey_west_regression(x, y, max_lag=5)
            correlation, correlation_p = stats.pearsonr(x, y)

            fig, ax = plt.subplots(figsize=(9.0, 5.4))
            ax.scatter(x, y, s=24, alpha=0.42, color=BLUE, edgecolor="none")
            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = regression["intercept"] + regression["slope"] * x_line
            ax.plot(x_line, y_line, color=GOLD, lw=2.2)
            ax.axhline(0, color=INK, lw=1.2, ls="--")
            ax.set(
                xlabel="Top-four average strength today, p0 (%)",
                ylabel="Same currencies’ average strength tomorrow, p1 (%)",
            )
            ax.set_title(
                "Today’s top-four strength versus next-day strength",
                loc="left", pad=34, fontsize=14, fontweight="bold"
            )
            ax.text(
                0.00, 1.015,
                f"OLS slope = {regression['slope']:+.3f} (Newey–West p = {regression['slope_p']:.3f}) · Pearson r = {correlation:+.3f}",
                transform=ax.transAxes, color=GREY, fontsize=9.5
            )
            sns.despine(ax=ax)
            plt.tight_layout()
            plt.show()

            regression_table = pd.DataFrame({
                "estimate": [regression["slope"], regression["slope_se"], regression["slope_p"], correlation, correlation_p],
            }, index=["p0→p1 slope", "Newey–West slope SE", "Slope p-value", "Pearson correlation", "Correlation p-value"])
            display(regression_table)
            """
        ),
        md(
            rf"""
            The fitted slope is **{slope:+.3f}** with $p={slope_p:.3f}$, and the correlation is **{correlation:+.3f}** with $p={correlation_p:.3f}$. The negative relationship is modest but statistically detectable: larger day-0 leader strength is associated with weaker day-1 performance. This is conditional mean reversion, even though the unconditional average remains close to zero.
            """
        ),
        code(
            """
            # Stability check: a 63-trading-day rolling estimate and the expanding mean
            stability = pd.DataFrame(index=results.index)
            stability["63-day rolling mean"] = results["p1_top4_avg"].rolling(63).mean()
            stability["Expanding mean"] = results["p1_top4_avg"].expanding(63).mean()

            fig, ax = plt.subplots(figsize=(11, 5.2))
            ax.plot(stability.index, stability["63-day rolling mean"], color=BLUE_LIGHT, lw=1.5,
                    label="63-day rolling mean")
            ax.plot(stability.index, stability["Expanding mean"], color=BLUE, lw=2.2,
                    label="Expanding mean")
            ax.axhline(0, color=INK, lw=1.2, ls="--")
            ax.set(
                xlabel="Signal date",
                ylabel="Average next-day strength (%)",
            )
            ax.set_title(
                "Stability of the estimated next-day effect",
                loc="left", pad=34, fontsize=14, fontweight="bold"
            )
            ax.text(
                0.00, 1.015,
                "Rolling estimates vary around small positive and negative values; the expanding estimate summarizes evidence available over time.",
                transform=ax.transAxes, color=GREY, fontsize=9.5
            )
            ax.legend(frameon=False, ncol=2, loc="upper left")
            sns.despine(ax=ax)
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            # Robustness: inspect future daily strength at 1, 2, and 5 trading-day horizons
            horizon_rows = []
            for horizon in [1, 2, 5]:
                values = []
                for position in range(len(groups) - horizon):
                    selected = groups.iloc[position]["top_4"]
                    values.append(currency_strength.iloc[position + horizon][selected].mean())
                values = np.asarray(values)
                low, high = block_bootstrap_mean_ci(values, seed=RANDOM_SEED + horizon)
                horizon_rows.append({
                    "horizon": horizon,
                    "mean": values.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "positive_share": (values > 0).mean(),
                    "observations": len(values),
                })

            horizon_results = pd.DataFrame(horizon_rows)

            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            errors = np.vstack([
                horizon_results["mean"] - horizon_results["ci_low"],
                horizon_results["ci_high"] - horizon_results["mean"],
            ])
            ax.errorbar(
                horizon_results["horizon"], horizon_results["mean"], yerr=errors,
                fmt="o", markersize=8, color=BLUE, ecolor=GOLD,
                elinewidth=2, capsize=5
            )
            ax.axhline(0, color=INK, lw=1.2, ls="--")
            ax.set(
                xticks=horizon_results["horizon"],
                xlabel="Future trading-day horizon",
                ylabel="Selected group’s future daily strength (%)",
            )
            ax.set_title(
                "Top-four signal across future daily horizons",
                loc="left", pad=34, fontsize=14, fontweight="bold"
            )
            ax.text(
                0.00, 1.015,
                "Points are means; error bars are 95% moving-block bootstrap intervals.",
                transform=ax.transAxes, color=GREY, fontsize=9.5
            )
            sns.despine(ax=ax)
            plt.tight_layout()
            plt.show()
            display(horizon_results.set_index("horizon"))
            """
        ),
        code(
            """
            # Data-source sensitivity: reconstruct strength from only the seven USD legs
            usd_spoke_strength = pd.DataFrame(
                0.0, index=pair_returns.index, columns=CURRENCIES
            )
            for currency in CURRENCIES:
                if currency != "USD":
                    usd_spoke_strength[currency] = directed_return(currency, "USD")
            usd_spoke_strength = usd_spoke_strength.sub(
                usd_spoke_strength.mean(axis=1), axis=0
            )

            usd_top4 = usd_spoke_strength.apply(
                lambda row: list(row.nlargest(4).index), axis=1
            )
            usd_records = []
            for position in range(len(usd_top4) - 1):
                selected = usd_top4.iloc[position]
                usd_records.append({
                    "p0": usd_spoke_strength.iloc[position][selected].mean(),
                    "p1": usd_spoke_strength.iloc[position + 1][selected].mean(),
                })
            usd_results = pd.DataFrame(usd_records)
            usd_regression = newey_west_regression(usd_results["p0"], usd_results["p1"])
            usd_ci = block_bootstrap_mean_ci(usd_results["p1"], seed=RANDOM_SEED + 100)

            usd_sensitivity = pd.Series({
                "next_day_mean_pct": usd_results["p1"].mean(),
                "continuation_rate": (usd_results["p1"] > 0).mean(),
                "bootstrap_ci_low_pct": usd_ci[0],
                "bootstrap_ci_high_pct": usd_ci[1],
                "p0_to_p1_slope": usd_regression["slope"],
                "newey_west_slope_p_value": usd_regression["slope_p"],
            }).to_frame("value")
            display(usd_sensitivity)
            """
        ),
        md(
            rf"""
            ## Takeaways

            ### What the evidence supports

            1. **The top/bottom symmetry is structural.** Equal-and-opposite group averages validate the zero-sum construction; they do not show that returns are random.
            2. **There is no evidence of momentum.** The top-four group’s mean next-day score was {mean:+.4f}%, with a 95% block-bootstrap interval of [{ci_low:+.4f}%, {ci_high:+.4f}%].
            3. **There is evidence of weak one-day mean reversion.** Continuation occurred on {positive_share:.1%} of days (sign-test $p={sign_p:.3f}$), and today’s top-four magnitude had a slope of {slope:+.3f} against tomorrow’s outcome ($p={slope_p:.3f}$).
            4. **The dependence survives a cleaner data construction.** Using only seven USD legs yields a slope of {usd_spoke_slope:+.3f} ($p={usd_spoke_slope_p:.3f}$), so redundant cross-rate noise does not explain the result by itself.
            5. **The pattern fades beyond day 1.** The descriptive two- and five-day-ahead daily-strength intervals span zero, so the reversal does not look persistent.
            6. **The conclusion is appropriately narrow.** The evidence contradicts the original “complete randomness” claim, but it does not establish a profitable strategy or a universal FX law.

            ### Decision

            **Reject the top-four ranking as a standalone momentum signal.** Keep the observed reversal as a research hypothesis only. Before deployment, require a frozen out-of-sample period, an investable portfolio mapping, and net-of-cost performance.

            ### Limitations

            - Yahoo Finance closes may not share an institutional FX fixing time across every cross.
            - The equal-weighted strength score reuses currencies across pairs and is not directly tradable.
            - No spreads, slippage, financing, or execution constraints are modeled.
            - Statistical non-significance is not proof of market efficiency or independence.
            - Exploring additional horizons after seeing the one-day result is descriptive and should not be treated as confirmatory evidence.

            ### Next steps

            - Freeze a later out-of-sample period before changing the feature definition.
            - Compare close-to-close, intraday, and institutional fixing-time data.
            - Test whether volatility, carry, macro regimes, or dispersion condition the signal.
            - Translate the index into an investable long/short construction and evaluate returns after costs.

            ---

            **Reproducibility:** run all cells from top to bottom. The frozen input is `data/fx_prices_2021-08-01_2026-08-01.csv`; deleting it triggers a fresh download for the configured date range.
            """
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
            "colab": {"provenance": [], "toc_visible": True},
            "case_study": {
                "title": "Do Today's Strongest Currencies Stay Strong Tomorrow?",
                "source_notebook": "C:/Users/Julio/Downloads/playground.ipynb",
                "build_period": [START_DATE, END_DATE],
                "tickers": tickers_by_pair,
            },
        },
    )
    nbf.write(notebook, NOTEBOOK_PATH)


def build_readme(summary: dict[str, object]) -> None:
    readme = dedent(
        rf"""
        # FX Currency Strength: Momentum or Mean Reversion?

        A reproducible Python case study testing whether the four strongest major currencies on one trading day continue to outperform on the next.

        ## Executive finding

        Across {len(summary['results']):,} next-day observations, the prior day’s top-four currencies produced an average next-day strength score of **{summary['mean']:+.4f}%**. The 95% moving-block bootstrap confidence interval was **[{summary['ci_low']:+.4f}%, {summary['ci_high']:+.4f}%]**. The estimated $p_0 \rightarrow p_1$ slope was **{summary['slope']:+.3f}** with a Newey–West $p$-value of **{summary['slope_p']:.3f}**.

        **Conclusion:** the sample rejects a momentum interpretation and shows a modest one-day mean-reversion relationship: continuation occurred on **{summary['positive_share']:.1%}** of days, and the $p_0 \rightarrow p_1$ slope was statistically negative. The effect is small, not directly investable, and not evidence of a profitable strategy.

        ## What the project demonstrates

        - Constructs all 28 unique pairs from eight major currencies.
        - Converts pair-level log returns into an equal-weighted currency-strength index.
        - Separates a zero-sum accounting identity from an empirical predictability test.
        - Uses a moving-block bootstrap, sign test, and Newey–West regression inference.
        - Audits triangular cross-rate coherence and repeats the test with only seven USD legs.
        - Communicates the result through distribution, relationship, stability, and robustness visuals.

        ## Repository contents

        - `fx_currency_strength_case_study.ipynb` — executed, reader-facing analysis.
        - `data/fx_prices_2021-08-01_2026-08-01.csv` — frozen Yahoo Finance input snapshot.
        - `requirements.txt` — Python dependencies.
        - `scripts/build_case_study.py` — deterministic notebook builder.

        ## Run locally

        ```bash
        python -m pip install -r requirements.txt
        jupyter nbconvert --execute --to notebook --inplace fx_currency_strength_case_study.ipynb
        ```

        The notebook reads the frozen CSV first. If it is absent, it downloads the configured range through `yfinance` and recreates the cache.

        ## Method in one line

        `28 FX closes` → `daily log returns` → `8 strength scores` → `top four on day t` → `same currencies on day t+1`

        ## Caveats

        The index is not directly investable; transaction costs are omitted; Yahoo Finance closes may not reflect a common institutional fixing time; and non-significance does not establish market efficiency or universal independence.

        ## Data source

        Yahoo Finance data accessed through [`yfinance`](https://ranaroussi.github.io/yfinance/). For research and portfolio demonstration only—not investment advice.
        """
    ).strip() + "\n"
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    requirements = dedent(
        """
        yfinance>=0.2.65
        pandas>=2.2
        numpy>=2.0
        scipy>=1.14
        matplotlib>=3.9
        seaborn>=0.13
        jupyter>=1.1
        nbclient>=0.10
        nbformat>=5.10
        """
    ).lstrip()
    (ROOT / "requirements.txt").write_text(requirements, encoding="utf-8")

    gitignore = dedent(
        """
        .ipynb_checkpoints/
        __pycache__/
        .venv/
        .notebook_runtime/
        .notebook_runtime_clean/
        """
    ).lstrip()
    (ROOT / ".gitignore").write_text(gitignore, encoding="utf-8")


def main() -> None:
    if CACHE_PATH.exists():
        prices = pd.read_csv(CACHE_PATH, index_col="Date", parse_dates=True)
        tickers_by_pair = {column: f"{column}=X" for column in prices.columns}
    else:
        prices, tickers_by_pair = download_prices()

    if prices.shape[1] != 28:
        raise RuntimeError(f"Expected 28 pair series, found {prices.shape[1]}")

    summary = analyse(prices)
    build_notebook(summary, tickers_by_pair)
    build_readme(summary)
    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"Data: {CACHE_PATH} ({prices.shape[0]:,} rows × {prices.shape[1]} pairs)")
    print(
        "Headline: "
        f"mean={summary['mean']:+.6f}%, "
        f"95% CI=[{summary['ci_low']:+.6f}, {summary['ci_high']:+.6f}], "
        f"slope={summary['slope']:+.6f}, p={summary['slope_p']:.6f}"
    )


if __name__ == "__main__":
    main()
