# Insurance Risk Analytics

Analysis of 18 months of insurance claim data (Feb 2014 – Aug 2015) for AlphaCare Insurance Solutions.

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
- Claims first appeared in November 2013
- Loss ratio increased from 0% to 14.02% over 18 months

## Data Cleaning

| Issue | Handling Method |
|-------|-----------------|
| Negative premiums/claims | Removed rows |
| Missing >60% | Dropped column |
| Missing 1-30% (numerical) | Filled with median |
| Missing 1-30% (categorical) | Filled with mode |
| Missing <1% | Dropped rows |


**Result:** 988,797 rows retained (98.9% of original), 0 missing values

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

# Launch notebook
jupyter notebook notebooks/01_eda.ipynb