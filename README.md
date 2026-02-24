# ⚽ Football League Prediction — End-to-End MLOps

Probabilistic prediction of **Premier League** and **La Liga** final standings using a Dixon-Coles Poisson model with Monte Carlo simulation. Built with a production-grade MLOps stack.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
│  football-data.org API → DVC versioned storage → Feature Store  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       TRAINING PIPELINE                         │
│  Prefect Orchestration → Dixon-Coles + Simulation → MLflow      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       SERVING LAYER                             │
│        FastAPI REST API  +  Streamlit Dashboard                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                      MONITORING LAYER                           │
│          Evidently AI drift detection + MLflow metrics           │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Stack

| Layer | Tool |
|---|---|
| Data versioning | DVC |
| Experiment tracking | MLflow |
| Orchestration | Prefect |
| API | FastAPI |
| Dashboard | Streamlit |
| Monitoring | Evidently AI |
| Containerisation | Docker + Docker Compose |

## 🚀 Quick Start

### 1. Clone & setup
```bash
git clone <your-repo>
cd football-mlops
cp .env.example .env
# Edit .env — add your football-data.org API key
```

### 2. Launch all services
```bash
docker compose up --build
```

### 3. Services available at:
| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prefect UI | http://localhost:4200 |

### 4. Run the full pipeline manually
```bash
docker compose exec app python pipelines/run_pipeline.py --leagues PL,PD
```

## 📁 Project Structure

```
football-mlops/
├── configs/               # League & model configuration
├── data/
│   ├── raw/               # DVC-tracked raw API data
│   ├── processed/         # DVC-tracked feature sets
│   └── external/          # Static reference data
├── src/
│   ├── data/              # Ingestion & validation
│   ├── features/          # Feature engineering
│   ├── models/            # Dixon-Coles + simulation
│   ├── api/               # FastAPI app
│   └── monitoring/        # Drift & accuracy tracking
├── pipelines/             # Prefect flow definitions
├── tests/                 # Unit & integration tests
├── docker/                # Dockerfiles
├── mlflow/                # MLflow artifact store config
└── notebooks/             # Exploratory notebooks
```

## 🔄 Pipeline Overview

1. **Ingest** — Pull latest fixtures & results from football-data.org
2. **Validate** — Great Expectations data quality checks
3. **Feature Engineer** — Rolling form, Elo ratings, home advantage
4. **Train** — Fit Dixon-Coles Poisson model, log to MLflow
5. **Simulate** — 10,000 Monte Carlo season simulations
6. **Promote** — Register best model in MLflow Model Registry
7. **Serve** — FastAPI serves predictions from registered model
8. **Monitor** — Weekly drift checks + prediction accuracy reports

## 📡 API Endpoints

```
GET  /predictions/{league}          → Current season standings probabilities
GET  /predictions/{league}/{team}   → Single team probability breakdown
GET  /health                        → Service health check
POST /predictions/refresh           → Trigger pipeline rerun
```

## 🔑 Environment Variables

```
FOOTBALL_DATA_API_KEY=your_key_here   # Free at football-data.org
MLFLOW_TRACKING_URI=http://mlflow:5000
PREFECT_API_URL=http://prefect:4200/api
```
