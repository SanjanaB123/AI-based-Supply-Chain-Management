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

# ── 1. Look up shared data ─────────────────────────────────────
data "terraform_remote_state" "foundation" {
  backend = "gcs"
  config = {
    bucket = var.tfstate_bucket
    prefix = "foundation"
  }
}

data "terraform_remote_state" "mcp" {
  backend = "gcs"
  config = {
    bucket = var.tfstate_bucket
    prefix = "mcp"
  }
}

# ── 2. IAM ─────────────────────────────────────────────────────
resource "google_secret_manager_secret" "anthropic_api_key" {
  secret_id = "anthropic-api-key-${var.environment}"
  project   = var.project_id
  replication { auto {} }
}

resource "google_secret_manager_secret_version" "anthropic_api_key" {
  secret      = google_secret_manager_secret.anthropic_api_key.id
  secret_data = var.anthropic_api_key
}

module "backend_iam" {
  source       = "../modules/iam-service-account"
  project_id   = var.project_id
  account_id   = "backend-${var.environment}"
  display_name = "Backend Service Account"
  roles = [
    "roles/logging.logWriter",
    "roles/artifactregistry.reader",
    "roles/cloudsql.client",
    "roles/secretmanager.secretAccessor",
  ]
}

# ── 3. Cloud Run Service ───────────────────────────────────────
module "backend" {
  source      = "../modules/cloud-run-service"
  project_id  = var.project_id
  region      = var.region
  name        = "backend-${var.environment}"
  image       = "${data.terraform_remote_state.foundation.outputs.artifact_repository_url}/backend:${var.backend_image_tag}"
  container_port = 8000
  
  service_account_email = module.backend_iam.email
  vpc_connector_id      = data.terraform_remote_state.foundation.outputs.vpc_connector_id
  db_instance_connection_name = data.terraform_remote_state.foundation.outputs.db_instance_connection_name
  
  cpu    = "2"
  memory = "8Gi"
  
  env_vars = [
    { name = "MCP_SERVER_URL", value = data.terraform_remote_state.mcp.outputs.uri },
    { name = "MCP_TRIGGER_RESTART", value = data.terraform_remote_state.mcp.outputs.latest_revision },
    { name = "ENVIRONMENT", value = var.environment },
    { name = "CLERK_JWKS_URL", value = var.clerk_jwks_url }
  ]
  
  secrets = [
    { name = "MONGO_URI",          secret_id = data.terraform_remote_state.foundation.outputs.mongo_uri_secret_id },
    { name = "ANTHROPIC_API_KEY",  secret_id = google_secret_manager_secret.anthropic_api_key.secret_id }
  ]
  
  is_public = true
}

output "uri" {
  value = module.backend.uri
}
