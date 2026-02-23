# AI-based-Supply-Chain-Management

Overview

This repository contains an end-to-end MLOps implementation of an AI-driven retail supply chain intelligence system. The project focuses on demand forecasting and inventory visibility at a store–product level, using production-oriented machine learning pipelines deployed on Google Cloud Platform (GCP).

The system is designed to forecast short-term demand, surface inventory risk signals (such as potential stockouts or overstock scenarios), and expose insights through both analytical dashboards and a natural language chat interface for store and warehouse managers. The emphasis of the project is on scalable architecture, reproducibility, and monitoring rather than isolated model experimentation.

Key Features

Daily demand forecasting at store–product granularity

Time-series feature engineering with lagged and rolling statistics

Automated training and batch inference pipelines

Model versioning and experiment tracking

Data and prediction drift monitoring

Manager-facing chat interface grounded in structured inventory and forecast data

Dashboarding for inventory health and forecast performance

Cloud-native deployment using GCP services

Dataset

The project uses a synthetic retail inventory forecasting dataset containing daily sales, inventory levels, pricing, promotions, holidays, and weather information across multiple stores and products. While synthetic, the dataset is structured to resemble real-world retail demand data and is used to validate system design and MLOps workflows.

Precomputed demand forecast fields present in the dataset are excluded from training to prevent data leakage.

Architecture Overview

The system follows a production-oriented MLOps architecture:

Storage: Google Cloud Storage for raw data, BigQuery for processed and feature tables

Pipelines: Scheduled data preprocessing, feature engineering, training, and inference

Model Management: Model registry and experiment tracking

Serving: Batch forecasts and optional online inference endpoints

Monitoring: Data drift and forecast performance tracking

Interfaces: Inventory dashboards and a tool-based chat assistant

Project Goals

Demonstrate end-to-end MLOps practices for time-series forecasting

Build a reproducible and monitorable ML system rather than a standalone model

Bridge forecasting outputs with operational decision support

Provide a foundation for future extensions such as reorder optimization and lead-time modeling

Out of Scope

Vendor and supplier lead-time optimization

Multi-echelon supply chain optimization

Automated procurement or ERP integration

Real-time point-of-sale ingestion

Tech Stack

Python

Google Cloud Platform (GCS, BigQuery, Vertex AI / Cloud Run)

Time-series and tree-based ML models

ML experiment tracking and model versioning

FastAPI for service endpoints

Dashboarding tools for visualization

Disclaimer

This project is intended for educational and system-design demonstration purposes. The dataset is synthetic and does not represent real commercial retail data.
