# Project Settings
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (e.g., dev, prod)"
  type        = string
  default     = "prod"
}

# Cloud SQL 
variable "db_password" {
  description = "Password for the Airflow Cloud SQL database user"
  type        = string
  sensitive   = true
}

variable "db_tier" {
  description = "Cloud SQL machine tier"
  type        = string
  default     = "db-f1-micro"
}

# Airflow secrets
variable "airflow_fernet_key" {
  description = "Fernet key for Airflow encryption"
  type        = string
  sensitive   = true
}

variable "airflow_jwt_secret" {
  description = "JWT secret for Airflow API auth"
  type        = string
  sensitive   = true
}

variable "mongo_uri" {
  description = "MongoDB connection string"
  type        = string
  sensitive   = true
}

variable "github_token" {
  description = "GitHub token for DVC push"
  type        = string
  sensitive   = true
}

variable "github_repo" {
  description = "GitHub repository (owner/repo)"
  type        = string
  default     = "SanjanaB123/AI-based-Supply-Chain-Management"
}

variable "gcs_bucket_name" {
  description = "GCS bucket for pipeline data"
  type        = string
  default     = "supply-chain-pipeline"
}

variable "airflow_admin_password" {
  description = "Password for the Airflow admin UI user"
  type        = string
  sensitive   = true
}

variable "airflow_image_tag" {
  description = "Docker image tag for the Airflow image"
  type        = string
  default     = "latest"
}
