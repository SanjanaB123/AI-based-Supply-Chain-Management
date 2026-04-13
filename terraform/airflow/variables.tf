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

variable "airflow_image_tag" {
  type    = string
  default = "latest"
}

variable "gcs_bucket_name" {
  type    = string
  default = "supply-chain-pipeline"
}

variable "github_repo" {
  type    = string
  default = "SanjanaB123/AI-based-Supply-Chain-Management"
}

variable "airflow_admin_password" {
  type      = string
  sensitive = true
  default   = "admin"
}
