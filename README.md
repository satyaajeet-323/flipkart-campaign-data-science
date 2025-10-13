## Flipkart Campaign Analysis

This repository contains utilities for analyzing Flipkart campaign performance, simple customer segmentation, and KPI calculations. It includes automated tests and a CI pipeline (Experiment 7) using open-source tools.

### Project Structure

```
data/
  flipkart_campaign.csv  # dataset (tracked outside git if using DVC)
src/
  campaign_analyzer.py   # load and summarize campaign performance
  customer_segmentation.py  # prepare features and cluster customers
  performance_metrics.py # CPA and KPI-over-time helpers
tests/
  test_*.py              # pytest unit tests
.github/workflows/ci-pipeline.yml  # GitHub Actions CI
```

### Installation

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Usage (basic)

```python
import pandas as pd
from src.campaign_analyzer import load_data, summarize_campaigns, top_campaigns_by_roi
from src.performance_metrics import calculate_cpa, kpi_over_time

df = load_data("data/flipkart_campaign.csv")
summary = summarize_campaigns(df)
top = top_campaigns_by_roi(df, top_n=5)
cpa = calculate_cpa(df)
kpi = kpi_over_time(df)
```

---

## Experiment 7: CI/CD Pipeline with Open Source Tools

**Aim**: CI/CD Pipeline with Open Source Tools

**Objective**: Automate testing, version checks, and build/deployment artifacts using GitHub Actions. Optionally pull data/models with DVC.

### Detailed Steps (Step-by-step)

1. **Create GitHub Actions workflow YAML**
   - The workflow lives at `.github/workflows/ci-pipeline.yml` and is already included.

2. **Add test, lint, and build jobs**
   - The workflow runs Ruff (lint), Black (format check), Pytest (tests), and builds sdist/wheel artifacts.

3. **Pull data/models with DVC (optional)**
   - If your dataset/models are stored with DVC remotes, configure a secret `DVC_REMOTE_URL` and appropriate credentials (e.g., S3 keys) in the repo settings. The `dvc pull` job runs automatically when the secret exists.

4. **Test pipeline on push to `main`**
   - Push a commit to `main` or open a Pull Request targeting `main`. The workflow triggers on both events.

### Open-Source Tools
- GitHub Actions (CI)
- DVC (data versioning, optional)

### Deliverables
- **Workflow YAML**: See `.github/workflows/ci-pipeline.yml`.
- **CI logs/screenshots**: After a run, open the GitHub repository → Actions → select the workflow run to view logs. Take screenshots of the successful jobs (Setup, DVC Pull if enabled, Tests, Build).
- **Conclusion**: With this pipeline, each change is automatically linted, formatted (checked), tested, and built. Optional DVC integration ensures data dependencies are materialized before tests/builds when configured.

### Running tests locally

```bash
pytest -q
```

### Formatting & Linting locally

```bash
ruff src tests
black .
```


