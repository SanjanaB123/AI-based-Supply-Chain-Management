# ======================================================================
# MAIN TERRAFORM CONFIG
# This file connects all modules together
# Think of it as: "Build networking FIRST, then database, then Cloud Run"
# ======================================================================

# ── Enable required GCP APIs ─────────────────────────────────────
# These are like "turning on" each GCP service before using it
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
  ])

  project = var.project_id
  service = each.value

  disable_dependent_services = false
  disable_on_destroy         = false
}

# ── 1. Networking (VPC + connector) ──────────────────────────────
module "networking" {
  source = "./modules/networking"

  project_id  = var.project_id
  region      = var.region
  environment = var.environment

  depends_on = [google_project_service.required_apis]
}

# ── 2. Cloud SQL (managed Postgres) ──────────────────────────────
module "cloud_sql" {
  source = "./modules/cloud-sql"

  project_id             = var.project_id
  region                 = var.region
  environment            = var.environment
  db_password            = var.db_password
  db_tier                = var.db_tier
  network_id             = module.networking.network_id
  private_vpc_connection = module.networking.private_vpc_connection

  depends_on = [module.networking]
}

# ── 3. Artifact Registry (Docker image storage) ─────────────────
module "artifact_registry" {
  source = "./modules/artifact-registry"

  project_id  = var.project_id
  region      = var.region
  environment = var.environment

  depends_on = [google_project_service.required_apis]
}

# ── 3b. GCS Bucket (pipeline data storage) ──────────────────────
module "gcs" {
  source = "./modules/gcs"

  project_id      = var.project_id
  region          = var.region
  environment     = var.environment
  gcs_bucket_name = var.gcs_bucket_name

  depends_on = [google_project_service.required_apis]
}

# ── 4. IAM (service account + permissions) ───────────────────────
module "iam" {
  source = "./modules/iam"

  project_id  = var.project_id
  environment = var.environment

  depends_on = [google_project_service.required_apis]
}

# ── 5. Secret Manager (all passwords and keys) ──────────────────
module "secret_manager" {
  source = "./modules/secret-manager"

  project_id  = var.project_id
  environment = var.environment

  db_password = var.db_password
  db_connection_string = format(
    "postgresql+psycopg2://airflow:%s@%s/airflow",
    var.db_password,
    module.cloud_sql.private_ip
  )
  airflow_fernet_key     = var.airflow_fernet_key
  airflow_jwt_secret     = var.airflow_jwt_secret
  mongo_uri              = var.mongo_uri
  github_token           = var.github_token
  airflow_admin_password = var.airflow_admin_password

  depends_on = [google_project_service.required_apis, module.cloud_sql]
}

# ── 6. Cloud Run (Airflow services + init job) ──────────────────
module "cloud_run" {
  source = "./modules/cloud-run"

  project_id  = var.project_id
  region      = var.region
  environment = var.environment

  airflow_image = "${module.artifact_registry.repository_url}/airflow:${var.airflow_image_tag}"

  service_account_email       = module.iam.service_account_email
  vpc_connector_id            = module.networking.vpc_connector_id
  db_instance_connection_name = module.cloud_sql.instance_connection_name
  gcs_bucket_name             = var.gcs_bucket_name
  github_repo                 = var.github_repo

  # Pass secret IDs so Cloud Run knows which secrets to read
  db_connection_string_secret_id = module.secret_manager.db_connection_string_secret_id
  fernet_key_secret_id           = module.secret_manager.fernet_key_secret_id
  jwt_secret_secret_id           = module.secret_manager.jwt_secret_secret_id
  mongo_uri_secret_id            = module.secret_manager.mongo_uri_secret_id
  github_token_secret_id             = module.secret_manager.github_token_secret_id
  airflow_admin_password_secret_id   = module.secret_manager.airflow_admin_password_secret_id

  depends_on = [
    module.networking,
    module.cloud_sql,
    module.iam,
    module.secret_manager,
    module.artifact_registry,
  ]
}
