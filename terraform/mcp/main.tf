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

# ── 1. Look up shared foundation data ──────────────────────────
data "terraform_remote_state" "foundation" {
  backend = "gcs"
  config = {
    bucket = var.tfstate_bucket
    prefix = "foundation"
  }
}

# ── 2. IAM ─────────────────────────────────────────────────────
module "mcp_iam" {
  source       = "../modules/iam-service-account"
  project_id   = var.project_id
  account_id   = "mcp-server-${var.environment}"
  display_name = "MCP Server Service Account"
  roles = [
    "roles/logging.logWriter",
    "roles/artifactregistry.reader",
    "roles/secretmanager.secretAccessor",
  ]
}

# ── 3. Cloud Run Service ───────────────────────────────────────
module "mcp_server" {
  source      = "../modules/cloud-run-service"
  project_id  = var.project_id
  region      = var.region
  name        = "mcp-server-${var.environment}"
  image       = "${data.terraform_remote_state.foundation.outputs.artifact_repository_url}/mcp:${var.mcp_image_tag}"
  container_port = 8001
  
  service_account_email = module.mcp_iam.email
  vpc_connector_id      = data.terraform_remote_state.foundation.outputs.vpc_connector_id
  
  cpu    = "2"
  memory = "8Gi"
  
  env_vars = [
    { name = "MLFLOW_TRACKING_URI", value = var.mlflow_tracking_uri },
    { name = "ENVIRONMENT", value = var.environment }
  ]

  secrets = [
    { name = "MONGO_URI", secret_id = data.terraform_remote_state.foundation.outputs.mongo_uri_secret_id }
  ]

  is_public = true
}

output "uri" {
  value = module.mcp_server.uri
}

output "latest_revision" {
  value = module.mcp_server.latest_revision
}
