# Insurance Risk Analytics

Analysis of 18 months of insurance claim data (Feb 2014 – Aug 2015) for AlphaCare Insurance Solutions.

## Overview

This project analyzes 18 months of car insurance data to identify low-risk segments, optimize pricing, and develop data-driven marketing strategies.

## Key Findings

| Metric | Value |
|--------|-------|
| Overall Loss Ratio | 100.86% |
| Total Premium | R60.5M |
| Total Claims | R61.0M |

### Geographic Risk
| Province | Loss Ratio |
|----------|------------|
| Gauteng | 118.47% |
| Western Cape | 105.05% |
| KwaZulu-Natal | 102.20% |
| Northern Cape | 28.27% |

### Vehicle Risk
- **Heavy Commercial:** 161.24% loss ratio
- **TOYOTA:** 81% of all claims (R49.4M)
- **Highest risk model:** TOYOTA QUANTUM 2.7 SESFIKILE 16s (R11.5M)

### Temporal Trends
- Loss ratio increased from 0% to 14.02% over 18 months

## Data Cleaning

| Issue | Handling |
|-------|----------|
| Negative premiums/claims | Removed rows |
| Missing >60% | Dropped column |
| Missing 1-30% (numerical) | Median imputation |
| Missing 1-30% (categorical) | Mode imputation |
| Missing <1% | Dropped rows |

**Result:** 988,797 rows retained (98.9% of original)

## A/B Hypothesis Testing

### Hypotheses Tested

| Hypothesis | KPI | Test | P-value | Decision |
|------------|-----|------|---------|----------|
| Province risk differences | Claim Frequency | Chi-squared | 0.0088 | **Reject H₀** |
| Province risk differences | Claim Severity | Mann-Whitney U | 0.6696 | Fail to reject |
| Zip code risk differences | Claim Frequency | Chi-squared | 0.6905 | Fail to reject |
| Zip code margin differences | Margin | Welch's t-test | 0.5924 | Fail to reject |
| Gender risk differences | Claim Frequency | Chi-squared | 0.5605 | Fail to reject |
| Gender risk differences | Claim Severity | Mann-Whitney U | 0.1287 | Fail to reject |

### Key Conclusions

**Reject H₀ for Province Claim Frequency (p = 0.0088)**
- Gauteng claim rate: 0.3189%
- Northern Cape claim rate: 0.1254%
- **Recommendation:** Increase Gauteng premiums 150-200%

**Fail to reject for all other hypotheses**
- No evidence for zip code-based pricing
- No evidence for gender-based pricing (male 0.2599% vs female 0.2085%, p = 0.5605)
- Gender analysis used Title imputation (Mr→Male, Mrs/Ms/Miss→Female) with 120,798 male and 5,275 female policies

### Statistical Methods

| KPI Type | Test Used |
|----------|-----------|
| Claim Frequency (categorical) | Chi-squared / Fisher's Exact |
| Claim Severity (numerical) | Mann-Whitney U |
| Margin (numerical, zero-inflated) | Welch's t-test |

## Project Structure

```
insurance-risk-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/                     # tracked by DVC
├── notebooks/
│   ├── 01_eda.ipynb

├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda_utils.py
├── reports/
│   └── final_report.md
├── tests/
├── .dvc/
├── .gitignore
├── dvc.yaml
├── requirements.txt
└── README.md
```

## Data Version Control (DVC)

This project uses DVC to version datasets, ensuring reproducibility and auditability.

### Data Versions

| Version | File | Description |
|---------|------|-------------|
| v1 | `MachineLearningRating_v3.txt` | Raw insurance data (pipe-delimited) |
| v2 | `insurance_data_cleaned.csv` | Cleaned data after EDA pipeline |

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/insurance-risk-analytics.git
cd insurance-risk-analytics

# Create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate

# Install
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Pull data from DVC remote
dvc pull



