variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "airflow_image" {
  description = "Full Docker image URL (e.g., us-central1-docker.pkg.dev/project/repo/image:tag)"
  type        = string
}

variable "service_account_email" {
  description = "Service account email for Cloud Run services"
  type        = string
}

variable "vpc_connector_id" {
  description = "VPC connector ID for Cloud SQL access"
  type        = string
}

variable "db_instance_connection_name" {
  description = "Cloud SQL instance connection name (project:region:instance)"
  type        = string
}

variable "gcs_bucket_name" {
  description = "GCS bucket for pipeline data"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository (owner/repo)"
  type        = string
}

# ── Secret IDs (from secret-manager module) ──────────────────────
variable "db_connection_string_secret_id" {
  type = string
}

variable "fernet_key_secret_id" {
  type = string
}

variable "jwt_secret_secret_id" {
  type = string
}

variable "mongo_uri_secret_id" {
  type = string
}

variable "github_token_secret_id" {
  type = string
}

variable "airflow_admin_password_secret_id" {
  type = string
}
