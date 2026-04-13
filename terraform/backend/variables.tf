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
variable "backend_image_tag" {
  type    = string
  default = "latest"
}

variable "clerk_jwks_url" {
  type        = string
  description = "The JWKS URL from Clerk for decoding JWTs"
}

variable "anthropic_api_key" {
  type        = string
  description = "Anthropic API key for the backend Claude integration"
  sensitive   = true
}
