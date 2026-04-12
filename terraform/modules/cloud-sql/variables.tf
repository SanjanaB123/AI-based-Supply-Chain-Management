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

variable "db_password" {
  description = "Database password for airflow user"
  type        = string
  sensitive   = true
}

variable "db_tier" {
  description = "Cloud SQL machine tier"
  type        = string
  default     = "db-f1-micro"
}

variable "network_id" {
  description = "VPC network ID (from networking module)"
  type        = string
}

variable "private_vpc_connection" {
  description = "Private VPC connection (must be created before Cloud SQL)"
}
