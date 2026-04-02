variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "db_password" {
  description = "Cloud SQL database password"
  type        = string
  sensitive   = true
}

variable "db_connection_string" {
  description = "Full SQLAlchemy connection string for Airflow"
  type        = string
  sensitive   = true
}

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
