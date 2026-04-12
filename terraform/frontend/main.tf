terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  backend "gcs" {}
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── 1. Look up data ────────────────────────────────────────────
data "terraform_remote_state" "foundation" {
  backend = "gcs"
  config = {
    bucket = var.tfstate_bucket
    prefix = "foundation"
  }
}

data "terraform_remote_state" "backend" {
  backend = "gcs"
  config = {
    bucket = var.tfstate_bucket
    prefix = "backend"
  }
}

# ── 2. IAM ─────────────────────────────────────────────────────
module "frontend_iam" {
  source       = "../modules/iam-service-account"
  project_id   = var.project_id
  account_id   = "frontend-${var.environment}"
  display_name = "Frontend Service Account"
  roles = [
    "roles/logging.logWriter",
    "roles/artifactregistry.reader",
  ]
}

# ── 3. Cloud Run Service ───────────────────────────────────────
module "frontend" {
  source      = "../modules/cloud-run-service"
  project_id  = var.project_id
  region      = var.region
  name        = "frontend-${var.environment}"
  image       = "${data.terraform_remote_state.foundation.outputs.artifact_repository_url}/frontend:${var.frontend_image_tag}"
  container_port = 80
  
  service_account_email = module.frontend_iam.email
  vpc_connector_id      = data.terraform_remote_state.foundation.outputs.vpc_connector_id
  
  env_vars = [
    { name = "VITE_API_URL", value = data.terraform_remote_state.backend.outputs.uri },
    { name = "VITE_CLERK_PUBLISHABLE_KEY", value = var.clerk_publishable_key }
  ]
  
  is_public = true
}

output "uri" {
  value = module.frontend.uri
}
