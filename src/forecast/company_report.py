"""Assembles the company report the web app renders: fetch, score, describe.

Thin by design -- `company.py` owns the data and `company_checks.py` owns every
threshold. This module only glues them together, names the gaps it found, and
guarantees the result is plain JSON.
"""

from __future__ import annotations

import datetime as dt
from typing import Callable

from src.forecast import company_checks as C
from src.forecast.company import CompanyDataError, fetch_company

DISCLAIMER = (
    "These are automated checks run against free vendor data, not research and not "
    "investment advice. Thresholds are fixed rules of thumb applied identically to "
    "every company, so they miss anything specific to this one — an industry where "
    "high debt is normal, a one-off charge, an accounting change. The fair value is "
    "a simple discounted-cash-flow sanity check on its stated assumptions, never a "
    "price target. Check the filings before risking money, and speak to a licensed "
    "adviser who knows your situation."
)


def _warnings(d: dict, sections: dict) -> list[str]:
    """Name the coverage gaps, so a low score is never mistaken for a bad company."""
    out = []
    if d.get("eps_growth_1y") is None and d.get("rev_growth_1y") is None:
        out.append("No analyst forecasts for this ticker — the Future axis scores 0 "
                   "because nothing could be checked, not because growth is poor.")
    if not (d.get("revenue") or d.get("net_income")):
        out.append("No annual income statement — Past performance is unscored. "
                   "Yahoo's coverage is patchy for ADRs, recent listings and trusts.")
    if not (d.get("equity") or d.get("total_assets")):
        out.append("No annual balance sheet — Financial health is unscored.")
    if sections["value"].get("dcf") is None:
        fin, px = d.get("financial_currency"), d.get("currency")
        if fin and px and fin.upper() != px.upper():
            out.append(f"Financials are reported in {fin} but the price is in {px}, "
                       f"so no fair value is estimated — converting them here would "
                       f"produce a confidently wrong number.")
        else:
            out.append("No positive free cash flow or share count, so no "
                       "discounted-cash-flow fair value could be estimated.")
    thin = [lab for key, lab in C.AXES if sections[key]["n_evaluable"] <= 2]
    if thin:
        out.append("Thin data on: " + ", ".join(thin) +
                   " — most checks there could not be evaluated.")
    return out


def build_report(symbol: str, refresh: bool = False,
                 fetcher: Callable[..., dict] | None = None, **kw) -> dict:
    """The full report payload for one ticker. Raises CompanyDataError on a bad symbol."""
    d = (fetcher or fetch_company)(symbol, refresh=refresh, **kw)

    sections = {
        "value": C.value_section(d),
        "future": C.future_section(d),
        "past": C.past_section(d),
        "health": C.health_section(d),
        "dividend": C.dividend_section(d),
        "management": C.management_section(d),
        "ownership": C.ownership_section(d),
    }

    return {
        "symbol": d.get("symbol") or str(symbol).strip().upper(),
        "as_of": d.get("as_of") or dt.date.today().isoformat(),
        "cached": bool(d.get("cached")),
        "overview": {
            "name": d.get("name"), "sector": d.get("sector"),
            "industry": d.get("industry"), "summary": d.get("summary"),
            "website": d.get("website"), "exchange": d.get("exchange"),
            "employees": d.get("employees"), "currency": d.get("currency"),
            "price": d.get("price"), "market_cap": d.get("market_cap"),
            "shares_outstanding": d.get("shares_outstanding"),
            "price_series": d.get("price_series") or [],
        },
        "snowflake": C.snowflake(sections),
        "sections": sections,
        "warnings": _warnings(d, sections),
        "disclaimer": DISCLAIMER,
    }


__all__ = ["build_report", "CompanyDataError", "DISCLAIMER"]
