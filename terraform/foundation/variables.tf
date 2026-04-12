variable "project_id" {
  type = string
}
variable "region" {
  type    = string
  default = "us-central1"
}
variable "environment" {
  type    = string
  default = "dev"
}
variable "tfstate_bucket" {
  type        = string
  description = "The GCS bucket name for Terraform state"
}

variable "db_password" {
  type      = string
  sensitive = true
}

variable "db_tier" {
  type    = string
  default = "db-f1-micro"
}

variable "airflow_fernet_key" {
  type      = string
  sensitive = true
}

variable "airflow_jwt_secret" {
  type      = string
  sensitive = true
}

variable "mongo_uri" {
  type      = string
  sensitive = true
}

variable "github_token" {
  type      = string
  sensitive = true
}
