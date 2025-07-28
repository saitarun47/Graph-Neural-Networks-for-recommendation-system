# GraphSAGE Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-20.10+-blue.svg)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/Airflow-2.7+-orange.svg)](https://airflow.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An Industrial-Grade, End-to-End Graph Neural Network (GNN) System for Movie Recommendations featuring Real-Time Inference, Automated MLOps, and Full-System Observability**

## Overview

**This project delivers a GraphSAGE-powered movie recommender with a full MLOps pipeline — all in a scalable, observable production architecture.**

- **API**: Fast, robust, and ready for millions of requests  
- **Real-time intelligence**: Observability with Prometheus/Grafana; alerts and retraining baked in  
- **Data science engineering**: GraphSAGE inductive GNN, not just off-the-shelf collaborative filtering  
- **MLOps automation**: Airflow DAGs, MLflow tracking, model governance, and Dockerized deployments  

## Why Graph Neural Networks for Recommendations?

Traditional methods (**matrix factorization, simple collaborative filtering**) can’t:  
- Adapt to unseen users/items (the “cold start” problem)  
- Exploit the full graph structure of user-item networks  
- Scale to big, ever-changing datasets  

**GraphSAGE GNNs** excel by:  
- Learning node embeddings that generalize (inductive inference!)  
- Capturing nuanced, high-order user-item relations  
- Adapting to new users/items dynamically  
- Exhibiting robustness and scalability for real-world traffic  

## Features at a Glance

- **Advanced GNN**: GraphSAGE model, 3 layers, residual connections, dropout, inductive node generalization  
- **MLOps Pipelines**: Orchestrated with Apache Airflow, traceable with MLflow, containers with Docker  
- **Observability**: System, API, and model monitoring with Prometheus metrics & Grafana dashboards  
- **API-first**: Async FastAPI REST, OpenAPI docs, and ready-to-deploy microservices  
- **Multi-Strategy Recs**: Supports classic, content-based, GNN-based, and hybrid recommendations  
- **Smart Automated Retraining**: Alerts trigger retraining when model performance drops  

## Under the Hood: GraphSAGE Architecture

**Model Highlights**  
- 3 GraphSAGE layers (mean aggregators, residual connections, batchnorm, dropout)  
- Handles cold-start with inductive inference (new users/movies, no retrain needed)  
- Decoder for link prediction, enabling next-item/user scoring  

**How It Works**  
1. **Graph Construction**: User-item interactions + features = heterogeneous graph  
2. **Neighborhood Sampling**: Efficient batched training  
3. **Feature Aggregation**: Mean pooling, message passing  
4. **Prediction**: Score any user/movie pair for personalized recommendations  

## Data & Pipeline

- **MovieLens 100K**: 943 users × 1682 movies, 100K ratings, full metadata  
- **Features**: User demographics + movie genres, titles, released dates  
- **Pipeline**:  
    1. Data ingestion → validation → feature extraction  
    2. Graph construction + PyTorch tensors  
    3. Batched GNN training & evaluation  
    4. Model checkpointing + logging  

## MLOps + Automation

- **Airflow DAGs**: Automates loading, preprocessing, training, evaluation, and deployment  
- **MLflow Tracking**: Logs experiments, metrics, and models  
- **Prometheus + Grafana**: Collects and visualizes metrics like API latency, model accuracy, and user engagement  
- **Alerting**: Grafana-based alerts auto-trigger retraining via webhook/Airflow if model performance drops  

## Quickstart

### Prerequisites
- Docker & Docker Compose  
- Python 3.8+  
- Minimum 8GB RAM recommended (full-stack with Neo4j, Grafana)  

### Run with Docker Compose

```bash
git clone https://github.com/saitarun47/Graph-Neural-Networks-for-recommendation-system.git
cd Graph-Neural-Networks-for-recommendation-system
docker-compose up -d
```


### Access Services

- **API Docs (Swagger)**: http://localhost:8000/docs  
- **Recommendation API**: http://localhost:8000  
- **Grafana Dashboard**: http://localhost:3000 
- **Prometheus**: http://localhost:9090  
- **Airflow**: http://localhost:8080 
- **Neo4j Browser**: http://localhost:7474  


### Trigger Training Pipeline

```bash
curl -X POST "http://localhost:8080/api/v1/dags/gnn_dag/dagRuns"   -H "Content-Type: application/json"   -u airflow:airflow   -d '{"conf": {}}'
```

## API Usage

### Get Recommendations

#### Existing Users

```bash
curl -X POST "http://localhost:8000/recommendations"   -H "Content-Type: application/json"   -d '{
    "user_id": 1,
    "num_recommendations": 10
  }'
```

#### New Users

```bash
curl -X POST "http://localhost:8000/recommendations"   -H "Content-Type: application/json"   -d '{
    "user_profile": {
      "preferred_genres": ["Action", "Sci-Fi"],
      "age": 25
    },
    "strategy": "hybrid",
    "num_recommendations": 10
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Monitoring & Observability

### Real-Time Metrics
- **Request Rate**: `rate(http_requests_total[5m])`  
- **Response Time**: `rate(graphsage_recommendations_duration_seconds_existing_gnn_model_sum[5m]) / rate(graphsage_recommendations_duration_seconds_existing_gnn_model_count[5m])`  
- **Model Performance**: `graphsage_model_prediction_score_existing{job="statsd-exporter"}`  
- **User Engagement**: `graphsage_recommendations_total_existing_gnn_model{job="statsd-exporter"}`  

### Dashboard Features
- **API Performance**: Real-time request monitoring  
- **Model Quality**: Prediction score trends  
- **User Analytics**: Recommendation effectiveness  
- **System Health**: Resource utilization  

## Development

### Local Development Setup

```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scriptsctivate

pip install -r requirements.txt
python main.py
```

### Project Structure

```
gnn/
├── dags/                # Airflow DAG definitions
├── data/                # Raw and processed datasets
├── models/              # Saved model checkpoints
├── monitoring/          # Prometheus & Grafana configs
├── src/                 # Core Python modules
│   └── components/      # GNN, data loaders, utils, etc.
├── docker-compose.yaml
├── main.py              # FastAPI application entrypoint
└── requirements.txt     # Python dependencies
```



## Key Features

- **End-to-end system**: Real production workflow
- **Modern ML Deployment**: Handles MLOps, monitoring, retraining, health checks  
- **Advanced graph ML**: Not just collaborative filtering — inductive graph learning!  
- **Enterprise engineering**: Automated scaling, containerization, alerting, and pipelines  
- **Data science + engineering**: Built, delivered, and observed a real recommendation engine  

