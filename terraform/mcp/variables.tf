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
variable "mcp_image_tag" {
  type    = string
  default = "latest"
}

variable "mlflow_tracking_uri" {
  type    = string
  default = ""
}

variable "gcs_bucket_name" {
  type        = string
  description = "GCS bucket name for model artifacts"
}
