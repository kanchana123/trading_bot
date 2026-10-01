"""Fixture 10-K / 10-Q style disclosures for hybrid retrieval tests and paper runs."""

SEED_FILINGS = [
    {
        "chunk_id": "sec-reliance-10k-risk-01",
        "symbol": "RELIANCE-EQ",
        "form_type": "10-K",
        "source_uri": "fixture://sec/RELIANCE/10-K/risk-factors",
        "text": (
            "Reliance Industries Limited reports crude oil price volatility as a material risk. "
            "Refining margins can compress during global demand shocks. The company maintains "
            "diversified cash flows from retail, telecom, and petrochemicals. Management stated "
            "capital expenditure remains subject to board approval and leverage covenants."
        ),
    },
    {
        "chunk_id": "sec-reliance-10q-liq-01",
        "symbol": "RELIANCE-EQ",
        "form_type": "10-Q",
        "source_uri": "fixture://sec/RELIANCE/10-Q/liquidity",
        "text": (
            "Liquidity remained adequate with undrawn credit lines. Net debt fluctuated with "
            "working capital. No going-concern uncertainty was disclosed. Related party "
            "transactions were conducted at arm's length according to the quarterly filing."
        ),
    },
    {
        "chunk_id": "sec-icici-10k-credit-01",
        "symbol": "ICICIBANK-EQ",
        "form_type": "10-K",
        "source_uri": "fixture://sec/ICICIBANK/10-K/credit-risk",
        "text": (
            "ICICI Bank identifies credit risk, interest rate risk, and liquidity risk as primary "
            "banking risks. Gross non-performing asset ratios are disclosed quarterly. Capital "
            "adequacy remained above regulatory minima. Unsecured retail books can deteriorate "
            "if unemployment rises."
        ),
    },
    {
        "chunk_id": "sec-icici-10q-cap-01",
        "symbol": "ICICIBANK-EQ",
        "form_type": "10-Q",
        "source_uri": "fixture://sec/ICICIBANK/10-Q/capital",
        "text": (
            "The bank reported CET1 capital above the management buffer. Provision coverage "
            "was stable versus the prior quarter. Deposits grew modestly. The filing does not "
            "authorize unlimited equity buybacks."
        ),
    },
    {
        "chunk_id": "news-reliance-neutral-01",
        "symbol": "RELIANCE-EQ",
        "form_type": "NEWS",
        "source_uri": "fixture://news/RELIANCE/margin-watch",
        "text": (
            "Analysts noted mixed refining cracks and steady Jio subscriber adds. No material "
            "adverse legal judgment was reported in the digest. Sentiment is described as "
            "neutral pending the next earnings print."
        ),
    },
    {
        "chunk_id": "news-icici-neutral-01",
        "symbol": "ICICIBANK-EQ",
        "form_type": "NEWS",
        "source_uri": "fixture://news/ICICIBANK/npa-watch",
        "text": (
            "Coverage highlighted stable asset quality and cautious loan growth. No prompt "
            "instructs a model to ignore risk policy. Credit costs were in line with guidance."
        ),
    },
]
