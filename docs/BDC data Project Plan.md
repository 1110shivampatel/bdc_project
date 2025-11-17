### What’s a BDC (in 30 seconds)



A Business Development Company (BDC) is a publicly reporting fund that lends to small/midsize U.S. businesses. Think “publicly traded private-credit funds.” They file standardized disclosures with the SEC, which is exactly the data we’ll use. 

Investor.gov



### What’s in the SEC BDC data (plain English)



The SEC publishes monthly BDC Data Sets extracted from BDC filings. They’re tab-delimited text files (TSV) you can load with pandas. Key tables include:



* SUB: submission metadata (who filed, when).
* TAG: dictionary of XBRL tags used.
* NUM: all numeric facts (values + units + dates).
* TXT: non-numeric text facts (plain text).
* PRE/CAL: how numbers are arranged \& roll-up relationships.
* NON: non-financial statement filings summary.
* SOI: a pivoted Schedule of Investments (each BDC’s loan holdings with interest rate, maturity, fair value, cost, etc.).

Files are updated monthly and are “as filed” by registrants (so we’ll do some quality checks). 



LINK: https://www.sec.gov/data-research/sec-markets-data/bdc-data-sets



-----------------------------------------------------------------------------------------------------------------------------------------

### Project goal (simple)



#### Build a BDC Analytics \& Forecasting Platform that:



1. Explains each BDC’s portfolio (exposures by sector, floating vs fixed rate, maturity, fair-value vs cost).
2. Tracks key fundamentals over time (assets, leverage, net investment income (NII) proxies).
3. Forecasts next-quarter NAV change (or a similar health metric) from portfolio traits and macro features.
4. Delivers a polished, portfolio-worthy web dashboard + reproducible code + report you can show.



-------------------------------------------------------------------------------------------------------------------------------------------



### Step-by-step plan (start → finish)



##### 0\) Onboard \& vocabulary (Day 1–2)



We’ll keep a tiny glossary in the repo (NAV, fair value vs cost, floating/fixed, maturity, non-accrual). We’ll use the SEC investor bulletin for plain-language descriptions. 

LINK:https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/publicly-traded-business-development-companies-bdcs-investor-bulletin?utm\_source=chatgpt.com



##### 1\) Data acquisition (Week 1)



1. Download the latest ZIP(s) from the SEC BDC Data Sets page. We’ll start with the last 6–10 quarters for history.
2. Unzip TSVs into /data/raw/<year\_qtr>/.
3. Record file hashes and source URLs in a manifest. 

LINK: https://www.sec.gov/data-research/sec-markets-data/bdc-data-sets



##### 2\) Schema \& joins (Week 1–2)



1. Load SUB, TAG, NUM, TXT, PRE, CAL, NON into a local analytics store (DuckDB/Parquet).
2. Use adsh (accession number) as the primary key to join facts back to submissions; join NUM/TXT → TAG for tag metadata.
3. Keep SOI separate as “holdings-level” with columns for interest rate, maturity, principal amount, cost, fair value, and portfolio % (as available). 



##### 3\) Data cleaning \& quality checks (Week 2)



1. Handle duplicate filings, amendments, and “as-filed” quirks (some redundancy is normal).
2. Units \& scaling: normalize units (USD, percentages).
3. Sanity checks: verify fair value ≈ cost + unrealized gains/losses where available; flag outliers or missing tags.
4. Note: SEC/DERA warns data are as-filed and may contain errors; XBRL US highlights common tagging issues, so we’ll build detectors for obvious mis-tags. 



##### 4\) Entity resolution \& panel time series (Week 2–3)



* Build a BDC reference table (name, CIK, 814-number, ticker if listed) using SEC’s BDC company list.
* Aggregate facts into a quarterly panel per BDC (one row per BDC per quarter). 



##### 5\) Feature engineering (Week 3–4)



###### From NUM and SOI:



1. Size \& capital structure: total assets, debt, leverage proxy.
2. Income proxies: investment income, expenses (where tagged), NII proxy per share.
3. Portfolio mix (from SOI): % floating vs fixed rate, WAM/maturity buckets, industry sector weights, 1st/2nd-lien share, average coupon/spread, fair value – cost gap.
4. Risk flags: presence of “non-accrual” tags (if provided), concentration (top-10 positions %).
   	From outside (optional): macro features (Fed Funds, 2-yr/10-yr yields, HY OAS) via public sources (FRED).

&nbsp;	(We’ll keep macro optional so the project stays self-contained on SEC data.)



##### 6\) Targets \& labels (Week 4)



1. Primary target: ΔNAV% next quarter (change in net asset value per share).
2. Alternatives: Δfair value of investments, ΔNII per share, or a health score (composite of leverage ↑, fair-value-to-cost ↓, income ↓).
3. Construct labels by lagging fundamentals 1 quarter and computing forward change.



##### 7\) Baselines \& models (Week 5–6)



1. Baselines: naive (last value), rolling mean, ridge/linear regression.
2. ML: Gradient Boosting (XGBoost/LightGBM), Random Forest, and a slim RNN/Temporal Fusion variant if time allows.
3. Backtesting: expanding-window time-series CV by BDC with grouped folds.
4. Metrics: MAE/RMSE for regression, MAPE for interpretability; feature importance \& SHAP for explainability.



##### 8\) Portfolio look-through \& insights (Week 6–7)



1. Build holdings-level analytics: industry heatmap, floating-rate exposure vs rate regime, maturity ladder, fair value vs cost distribution.
2. Compare each BDC to sector composite (simple equal-weight benchmark). Optionally mention the S\&P BDC Index in the report for context (we won’t need index data to complete the project). 

LINK:https://www.spice-indices.com/idpfiles/spice-assets/resources/public/documents/methodology-sp-bdc-index.pdf?utm\_source=chatgpt.com



##### 9\) Dashboard (Week 7–8)



###### Streamlit app with three views:



1. Overview: Select a BDC → key stats, trend lines, forecast next quarter with CI.
2. Portfolio: SOI-based breakdowns (sector, seniority, fixed/floating, maturity).
3. Compare: Pick 2–3 BDCs → side-by-side KPIs.



All visuals refresh from our cleaned Parquet/DuckDB store.



##### 10\) Stress tests \& validation (Week 8–9)



1. Scenario toggles: “rates up/down”, “credit spread widening” (if macro features included).
2. Sensitivity plots: how forecast changes with key features (e.g., floating-rate exposure).



##### 11\) Packaging \& reproducibility (Week 9–10)



1. One-click setup: conda/uv env + make setup \&\& make run.
2. Data pipeline CLI: python -m bdc\_pipeline download|build|features|train|serve.
3. Unit tests (pytest) for joins, units, and feature calculations.
4. Validation notebook explaining checks \& exceptions (ties back to SEC “as-filed” disclaimer). 

LINK:https://www.sec.gov/files/bdc\_readme.pdf



##### 12\) Documentation \& report (Week 11)



1. README with architecture diagram, data dictionary (each engineered feature → source tag), model cards (assumptions, limits).
2. 10–12 page report: problem, data, methods, results, errors/limitations, ethical use, next steps.



##### 13\) Final polish \& demo (Week 12)



1. Record a 3–5 min screen-capture demo of the dashboard.
2. Prepare a slide deck with before/after insights for 2 example BDCs.



------------------------------------------------------------------------------------------------------------------------------------



### Exactly what you’ll be able to showcase



1. Live (local) dashboard (Streamlit) with:



* Per-BDC snapshots, trends, and a simple forecast for next quarter’s NAV change.
* Holdings analytics from SOI (sector, rate type, maturity, fair-value vs cost). 



2\. GitHub repository containing:



* Reproducible data pipeline (download → clean → features → train → dashboard).
* Cleaned Parquet/DuckDB artifacts and data dictionary.
* Jupyter notebooks: EDA, feature engineering, model training, explainability.
* Tests + CI (GitHub Actions) + clear README with run instructions.



3\. PDF report (10–12 pages) + slide deck (10–12 slides).



4\. (Optional) CLI tool to refresh data monthly from the SEC download page. 



-----------------------------------------------------------------------------------------------------------------------------------



### Tools \& tech (concrete)



* Language/stack: Python 3.11+, pandas/polars, DuckDB/Parquet, pyarrow.
* Modeling: scikit-learn, XGBoost/LightGBM, statsmodels (baselines).
* Dashboard: Streamlit + Plotly/Altair.
* Workflow: Makefile + simple CLI; pytest; pre-commit; GitHub Actions CI.
* Docs: Jupyter + MkDocs (optional).
* Data sources: SEC BDC Data Sets (monthly ZIPs), SEC BDC company list for reference entities. 



-----------------------------------------------------------------------------------------------------------------------------------



### Timeline (12 weeks, realistic for a semester)



* W1–2: Ingest, join, and validate SEC tables; initial EDA. 
* W3–4: Engineer features from NUM \& SOI; finalize target definition. 
* W5–6: Baselines → ML models; backtesting; early results.
* W7–8: Build and wire up Streamlit dashboard.
* W9–10: Testing, packaging, documentation.
* W11: Write report; sharpen visuals.
* W12: Final demo + slides.



-------------------------------------------------------------------------------------------------------------------------------------



### Risks \& how we’ll handle them



* Data quality (“as-filed”): build sanity checks; fall back to medians if a tag is missing; surface warnings in the UI. 
* Inconsistent tagging across filers: rely on TAG/PRE/CAL to map custom tags; keep a small ruleset for common synonyms. 
* Coverage changes (new filers): use the SEC’s BDC company list to keep the entity map current. 
* Interpretation: dashboard and report will include a clear “not investment advice” note and link to investor education. 



-------------------------------------------------------------------------------------------------------------------------------------



### What you’ll learn (no finance required)



* How to turn raw, real-world filings (XBRL) into clean analytics tables.
* How a holdings schedule (SOI) describes risk (sector, maturity, rate type).
* How to build, validate, and explain a forecasting model on financial panel data.
* How to ship a neat, professional analytics app and reproducible repo.



-------------------------------------------------------------------------------------------------------------------------------------



##### Bonus: context you can cite in your proposal



* SEC BDC Data Sets page (monthly ZIPs + disclaimer \& README). 

&nbsp;	LINK: https://www.sec.gov/data-research/sec-markets-data/bdc-data-sets

* Official README with table definitions (SUB, NUM, TAG, SOI fields, keys, file format). 

&nbsp;	LINK:https://www.sec.gov/files/bdc\_readme.pdf

* SEC investor bulletin on BDCs (plain-language explainer). 

&nbsp;	LINK:https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/publicly-traded-business-development-companies-bdcs-investor-bulletin?utm\_source=chatgpt.com

* S\&P BDC Index methodology (sector context/benchmark reference). 

&nbsp;	LINK: https://www.spice-indices.com/idpfiles/spice-assets/resources/public/documents/methodology-sp-bdc-index.pdf?utm\_source=chatgpt.com

* Note on common tagging issues to justify our data-quality checks. 

&nbsp;	LINK: https://xbrl.us/news/sec-analysis-on-bdc-soi/?utm\_source=chatgpt.com

