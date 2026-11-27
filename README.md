# TellCo Analytics

[![CI](https://github.com/sskanji2011-max/tellco-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/sskanji2011-max/tellco-analytics/actions)

Live dashboard: https://tellco-analytics-sk.streamlit.app

User overview, engagement, experience and satisfaction analytics for the TellCo xDR dataset
(Nexthikes IT Solutions brief).

## Layout
| Path | Purpose |
|---|---|
| `src/tellco/` | Installable package: `data` (cleaning), `features` (aggregation + feature store), `engagement`, `experience`, `satisfaction`, `pipeline` (end-to-end + MLflow) |
| `app/dashboard.py` | Streamlit dashboard |
| `tests/` | pytest suite (synthetic data, ~96 % coverage) |
| `Dockerfile` | Builds the dashboard image |
| `.github/workflows/ci.yml` | Lint, tests (Py 3.10/3.11) and Docker build on every push |
| `reports/` | Slide deck and exported results |

## Quick start
```bash
pip install -e ".[dashboard,dev]"
tellco-run --input data/raw/telcom_data.xlsx          # writes data/processed/* and logs MLflow runs
streamlit run app/dashboard.py                        # dashboard on http://localhost:8501
python -m mlflow ui --backend-store-uri sqlite:///mlflow_tellco.db     # model tracking UI on http://localhost:5000
pytest                                                # unit tests + coverage
```

## Docker
```bash
docker build -t tellco-analytics .
docker run --rm -v $PWD/data:/app/data tellco-analytics tellco-run --input data/raw/telcom_data.xlsx
docker run --rm -p 8501:8501 -v $PWD/data:/app/data tellco-analytics
```

## Method notes
* Missing numerics: column mean; text: mode; handsets: `undefined`. Outliers (IQR rule) are replaced with the mean of non-outliers.
* Engagement score = Euclidean distance to the least engaged k-means cluster (min-max scaled metrics).
* Experience score = Euclidean distance to the worst experience cluster (standardised metrics).
* Both distances are rescaled to 0-1; satisfaction = their average.
* Feature store: `tellco.features.save_features / load_features` (`data/processed/feature_store/features.csv`).
