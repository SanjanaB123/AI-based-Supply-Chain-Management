# ── 1. Providers & Backend ──────────────────────────────────────
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

# ── 2. Enable Required APIs ──────────────────────────────────────
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "vpcaccess.googleapis.com",
    "compute.googleapis.com",
    "servicenetworking.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "storage.googleapis.com",
  ])

  project = var.project_id
  service = each.value

  disable_dependent_services = false
  disable_on_destroy         = false
}

# ── 3. Networking (VPC + connector) ──────────────────────────────
module "networking" {
  source = "../modules/networking"

  project_id  = var.project_id
  region      = var.region
  environment = var.environment

  depends_on = [google_project_service.required_apis]
}

# ── 4. Cloud SQL (managed Postgres) ──────────────────────────────
module "cloud_sql" {
  source = "../modules/cloud-sql"

  project_id             = var.project_id
  region                 = var.region
  environment            = var.environment
  db_password            = var.db_password
  db_tier                = var.db_tier
  network_id             = module.networking.network_id
  private_vpc_connection = module.networking.private_vpc_connection

  depends_on = [module.networking]
}

# ── 5. Artifact Registry ────────────────────────────────────────
module "artifact_registry" {
  source = "../modules/artifact-registry"

  project_id  = var.project_id
  region      = var.region
  environment = var.environment

  depends_on = [google_project_service.required_apis]
}

# ── 6. Shared Secretariat (Secret Manager Setup) ─────────────────
module "secret_manager" {
  source = "../modules/secret-manager"

  project_id  = var.project_id
  environment = var.environment

  db_password = var.db_password
  db_connection_string = format(
    "postgresql+psycopg2://airflow:%s@%s/airflow",
    var.db_password,
    module.cloud_sql.private_ip
  )
  airflow_fernet_key = var.airflow_fernet_key
  airflow_jwt_secret = var.airflow_jwt_secret
  mongo_uri          = var.mongo_uri
  github_token       = var.github_token

  depends_on = [google_project_service.required_apis, module.cloud_sql]
}
