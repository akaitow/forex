from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "fx_prices_2021-08-01_2026-08-01.csv"
NOTEBOOK_PATH = ROOT / "fx_currency_strength_case_study.ipynb"
ASSET_DIR = ROOT / "assets"

CURRENCIES = ["EUR", "USD", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD"]
RANDOM_SEED = 42


def calculate_results(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_returns = prices.pct_change(fill_method=None).mul(100).dropna(how="any")
    strength = pd.DataFrame(0.0, index=pair_returns.index, columns=CURRENCIES)

    for pair in pair_returns.columns:
        base, quote = pair[:3], pair[3:]
        strength[base] = strength[base] + pair_returns[pair] / 7
        strength[quote] = strength[quote] - pair_returns[pair] / 7

    assert pair_returns.shape[1] == 28
    assert np.allclose(strength.sum(axis=1), 0, atol=1e-10)

    top4_by_day = strength.apply(lambda row: list(row.nlargest(4).index), axis=1)
    records = []

    for position in range(len(top4_by_day) - 1):
        selected = top4_by_day.iloc[position]
        records.append(
            {
                "date": strength.index[position],
                "next_date": strength.index[position + 1],
                "top_4": ", ".join(selected),
                "p0_top4_avg": strength.iloc[position][selected].mean(),
                "top4_next_day_avg": strength.iloc[position + 1][selected].mean(),
            }
        )

    return strength, pd.DataFrame(records).set_index("date")


def bootstrap_mean_ci(values: pd.Series, simulations: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    array = values.to_numpy()
    bootstrap_means = np.array(
        [rng.choice(array, size=len(array), replace=True).mean() for _ in range(simulations)]
    )
    return tuple(np.quantile(bootstrap_means, [0.025, 0.975]))


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook() -> None:
    prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    _, results = calculate_results(prices)
    outcome = results["top4_next_day_avg"]
    ci_low, ci_high = bootstrap_mean_ci(outcome)
    regression = stats.linregress(results["p0_top4_avg"], outcome)
    r_squared = regression.rvalue**2

    cells = [
        markdown(
            f"""
            # Do Today’s Strongest Currencies Stay Strong Tomorrow?

            ## TL;DR

            Across **{len(results):,} trading days**, the four strongest currencies on day 0 averaged **{outcome.mean():+.4f}%** on day 1. The 95% bootstrap confidence interval was **[{ci_low:+.4f}%, {ci_high:+.4f}%]**, which includes zero.

            A linear model found a small negative relationship, but day‑0 strength explained only **{r_squared:.1%}** of day‑1 variation.

            **Conclusion:** the ranking did not produce a useful continuation signal. At most, the sample suggests weak one-day mean reversion.
            """
        ),
        markdown(
            """
            ## Context & Methods

            ### Purpose

            Test whether the four strongest currencies today tend to remain strong tomorrow.

            ### Problem

            FX prices are quoted as pairs. To compare currencies individually, each pair return is converted into a contribution for its base currency and the opposite contribution for its quote currency.

            ### Hypothesis

            - **Null:** the period average of `top4_next_day_avg` is zero and day‑0 strength does not explain day‑1 strength.
            - **Momentum:** the next-day average is positive.
            - **Mean reversion:** the next-day average is negative.

            ### Method

            1. Calculate daily percentage changes for 28 pairs built from eight major currencies.
            2. Average each currency’s signed contribution across its seven pairs.
            3. Select the four strongest currencies each day.
            4. Measure those same four currencies on the next trading day.
            5. Study the period distribution, mean confidence interval, and day‑0/day‑1 regression.

            Because the eight strength scores sum to zero, the bottom-four average is the mirror of the top-four average. Only the top-four result is needed.
            """
        ),
        code(
            """
            # Setup and load the frozen dataset
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from scipy import stats

            CURRENCIES = ["EUR", "USD", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD"]
            DATA_PATH = Path("data/fx_prices_2021-08-01_2026-08-01.csv")
            ASSET_DIR = Path("assets")
            RANDOM_SEED = 42

            BLUE = "#315F8C"
            LIGHT_BLUE = "#BDD0E2"
            GOLD = "#C58A1B"
            INK = "#263442"
            GREY = "#788794"
            GRID = "#DCE3E8"

            ASSET_DIR.mkdir(exist_ok=True)
            prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
            print(f"{prices.shape[0]:,} dates × {prices.shape[1]} currency pairs")
            prices.head(3)
            """
        ),
        code(
            """
            # Build daily currency-strength scores
            pair_returns = prices.pct_change(fill_method=None).mul(100).dropna(how="any")
            currency_strength = pd.DataFrame(
                0.0, index=pair_returns.index, columns=CURRENCIES
            )

            for pair in pair_returns.columns:
                base, quote = pair[:3], pair[3:]
                currency_strength[base] = currency_strength[base] + pair_returns[pair] / 7
                currency_strength[quote] = currency_strength[quote] - pair_returns[pair] / 7

            assert pair_returns.shape[1] == 28
            assert np.allclose(currency_strength.sum(axis=1), 0, atol=1e-10)

            currency_strength.head(3)
            """
        ),
        code(
            """
            # Select each day’s top four and measure the same currencies one day later
            top4_by_day = currency_strength.apply(
                lambda row: list(row.nlargest(4).index), axis=1
            )

            records = []
            for position in range(len(top4_by_day) - 1):
                selected = top4_by_day.iloc[position]
                records.append({
                    "date": currency_strength.index[position],
                    "next_date": currency_strength.index[position + 1],
                    "top_4": ", ".join(selected),
                    "p0_top4_avg": currency_strength.iloc[position][selected].mean(),
                    "top4_next_day_avg": currency_strength.iloc[position + 1][selected].mean(),
                })

            next_day_results = pd.DataFrame(records).set_index("date")
            next_day_results.head()
            """
        ),
        markdown("## Results"),
        code(
            """
            # Period-level summary of the daily top-four next-day group averages
            outcome = next_day_results["top4_next_day_avg"]
            rng = np.random.default_rng(RANDOM_SEED)
            bootstrap_means = np.array([
                rng.choice(outcome.to_numpy(), size=len(outcome), replace=True).mean()
                for _ in range(10_000)
            ])
            ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])

            regression = stats.linregress(
                next_day_results["p0_top4_avg"], outcome
            )

            summary = pd.Series({
                "observations": len(outcome),
                "period_mean_pct": outcome.mean(),
                "period_median_pct": outcome.median(),
                "ci_95_low_pct": ci_low,
                "ci_95_high_pct": ci_high,
                "regression_slope": regression.slope,
                "regression_r_squared": regression.rvalue ** 2,
                "regression_p_value": regression.pvalue,
            })
            summary.to_frame("value")
            """
        ),
        code(
            """
            # Visual 1: distribution and uncertainty around the period mean
            fig = plt.figure(figsize=(10, 6.5), facecolor="white")
            grid = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.06)
            ax = fig.add_subplot(grid[0])
            interval_ax = fig.add_subplot(grid[1], sharex=ax)

            ax.hist(outcome, bins=38, color=LIGHT_BLUE, edgecolor=BLUE, linewidth=0.7)
            ax.axvline(0, color=INK, linestyle="--", linewidth=1.3, label="Zero")
            ax.axvline(outcome.mean(), color=GOLD, linewidth=2.2, label="Period mean")
            ax.set_ylabel("Trading days")
            ax.set_title(
                "Top-four next-day averages are centered close to zero",
                loc="left", fontsize=15, fontweight="bold", color=INK, pad=22
            )
            ax.text(
                0, 1.01,
                f"Distribution of {len(outcome):,} daily group averages; selected period: "
                f"{outcome.index.min().date()} to {outcome.index.max().date()}",
                transform=ax.transAxes, color=GREY, fontsize=10
            )
            ax.legend(frameon=False, ncol=2, loc="upper left")
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", labelbottom=False)

            interval_ax.errorbar(
                outcome.mean(), 0,
                xerr=[[outcome.mean() - ci_low], [ci_high - outcome.mean()]],
                fmt="o", color=BLUE, ecolor=GOLD, elinewidth=3, capsize=6, markersize=7
            )
            interval_ax.axvline(0, color=INK, linestyle="--", linewidth=1.3)
            interval_ax.set_ylim(-0.9, 0.9)
            interval_ax.set_yticks([])
            interval_ax.set_xlabel("Top-four next-day average (%)")
            interval_ax.text(
                0.0, 0.86,
                f"Mean {outcome.mean():+.4f}% · 95% CI [{ci_low:+.4f}%, {ci_high:+.4f}%]",
                transform=interval_ax.transAxes, color=GREY, fontsize=10
            )
            interval_ax.spines[["top", "right", "left"]].set_visible(False)
            interval_ax.grid(axis="x", color=GRID, linewidth=0.8)

            plt.tight_layout()
            plt.savefig(ASSET_DIR / "top4_next_day_distribution.png", dpi=180, bbox_inches="tight")
            plt.show()
            """
        ),
        code(
            """
            # Visual 2: linear relationship between day 0 and day 1
            x = next_day_results["p0_top4_avg"].to_numpy()
            y = next_day_results["top4_next_day_avg"].to_numpy()
            x_line = np.linspace(x.min(), x.max(), 250)
            y_line = regression.intercept + regression.slope * x_line

            residuals = y - (regression.intercept + regression.slope * x)
            residual_std = np.sqrt(np.sum(residuals ** 2) / (len(x) - 2))
            x_centered_sum = np.sum((x - x.mean()) ** 2)
            mean_se = residual_std * np.sqrt(
                1 / len(x) + (x_line - x.mean()) ** 2 / x_centered_sum
            )
            critical_t = stats.t.ppf(0.975, df=len(x) - 2)

            fig, ax = plt.subplots(figsize=(10, 5.8), facecolor="white")
            ax.scatter(x, y, s=20, alpha=0.32, color=BLUE, edgecolor="none")
            ax.plot(x_line, y_line, color=GOLD, linewidth=2.3)
            ax.fill_between(
                x_line,
                y_line - critical_t * mean_se,
                y_line + critical_t * mean_se,
                color=GOLD, alpha=0.18, linewidth=0
            )
            ax.axhline(0, color=INK, linestyle="--", linewidth=1.2)
            ax.set_xlabel("Top-four average on day 0 (%)")
            ax.set_ylabel("Same currencies’ average on day 1 (%)")
            ax.set_title(
                "Day-0 strength explains little of the next day",
                loc="left", fontsize=15, fontweight="bold", color=INK, pad=22
            )
            p_text = "< 0.001" if regression.pvalue < 0.001 else f"= {regression.pvalue:.3f}"
            ax.text(
                0, 1.01,
                f"Slope = {regression.slope:+.3f} · R² = {regression.rvalue ** 2:.1%} · p {p_text}",
                transform=ax.transAxes, color=GREY, fontsize=10
            )
            ax.grid(color=GRID, linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)

            plt.tight_layout()
            plt.savefig(ASSET_DIR / "p0_p1_regression.png", dpi=180, bbox_inches="tight")
            plt.show()
            """
        ),
        markdown(
            f"""
            ## Takeaways

            - The period mean was **{outcome.mean():+.4f}%** and its 95% confidence interval included zero.
            - The regression slope was **{regression.slope:+.3f}**, but $R^2$ was only **{r_squared:.1%}**.
            - The data do not support a useful continuation signal.
            - The small negative slope suggests weak mean reversion, but its explanatory power is too low to treat as a standalone forecast.

            ### Limitation

            This is an exploratory strength index, not a backtested trading strategy. It does not include spreads, slippage, financing, or out-of-sample testing.
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
        },
    )
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Built {NOTEBOOK_PATH}")
    print(
        f"mean={outcome.mean():+.6f}, ci=[{ci_low:+.6f}, {ci_high:+.6f}], "
        f"slope={regression.slope:+.6f}, r2={r_squared:.6f}"
    )


if __name__ == "__main__":
    build_notebook()
