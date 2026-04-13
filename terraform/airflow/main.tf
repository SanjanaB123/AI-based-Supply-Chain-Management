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

locals {
  airflow_image = "${data.terraform_remote_state.foundation.outputs.artifact_repository_url}/airflow:${var.airflow_image_tag}"

  common_env = [
    { name = "AIRFLOW__CORE__EXECUTOR",                    value = "LocalExecutor" },
    { name = "AIRFLOW__CORE__AUTH_MANAGER",                value = "airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager" },
    { name = "AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION", value = "true" },
    { name = "AIRFLOW__CORE__LOAD_EXAMPLES",               value = "false" },
    { name = "AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK",    value = "true" },
    { name = "PYTHONPATH",                                  value = "/opt/airflow" },
    { name = "PARAMS_PATH",                                 value = "/opt/airflow/params.yaml" },
    { name = "GCS_BUCKET_NAME",                             value = var.gcs_bucket_name }, # Could also lookup from state if moved
    { name = "GITHUB_REPO",                                 value = var.github_repo },
  ]

  common_secrets = [
    { name = "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN", secret_id = data.terraform_remote_state.foundation.outputs.db_connection_string_secret_id },
    { name = "AIRFLOW__CORE__FERNET_KEY",           secret_id = data.terraform_remote_state.foundation.outputs.fernet_key_secret_id },
    { name = "AIRFLOW__API_AUTH__JWT_SECRET",       secret_id = data.terraform_remote_state.foundation.outputs.jwt_secret_secret_id },
    { name = "MONGO_URI",                           secret_id = data.terraform_remote_state.foundation.outputs.mongo_uri_secret_id },
    { name = "GITHUB_TOKEN",                        secret_id = data.terraform_remote_state.foundation.outputs.github_token_secret_id },
  ]
}

# ── 2. IAM ─────────────────────────────────────────────────────
module "airflow_iam" {
  source       = "../modules/iam-service-account"
  project_id   = var.project_id
  account_id   = "airflow-cloudrun-${var.environment}"
  display_name = "Airflow Service Account"
  roles = [
    "roles/logging.logWriter",
    "roles/artifactregistry.reader",
    "roles/cloudsql.client",
    "roles/secretmanager.secretAccessor",
    "roles/storage.objectAdmin",
  ]
}

# ── 3. Airflow Services ────────────────────────────────────────
module "airflow_webserver" {
  source      = "../modules/cloud-run-service"
  project_id  = var.project_id
  region      = var.region
  name        = "airflow-webserver-${var.environment}"
  image       = local.airflow_image
  container_port = 8080
  
  service_account_email       = module.airflow_iam.email
  vpc_connector_id            = data.terraform_remote_state.foundation.outputs.vpc_connector_id
  db_instance_connection_name = data.terraform_remote_state.foundation.outputs.db_instance_connection_name
  
  command = ["/opt/airflow/entrypoint.sh"]
  args    = ["api-server", "--port", "8080"]
  
  memory = "4Gi"
  cpu    = "1"
  cpu_idle = false

  env_vars = concat(local.common_env, [
    { name = "AIRFLOW__CORE__EXECUTION_API_SERVER_URL", value = "http://localhost:8080/execution/" },
    { name = "AIRFLOW__WEBSERVER__WORKERS",           value = "1" }
  ])
  secrets = local.common_secrets
  
  is_public = true
}

module "airflow_scheduler" {
  source      = "../modules/cloud-run-service"
  project_id  = var.project_id
  region      = var.region
  name        = "airflow-scheduler-${var.environment}"
  image       = local.airflow_image
  container_port = 8080
  
  service_account_email       = module.airflow_iam.email
  vpc_connector_id            = data.terraform_remote_state.foundation.outputs.vpc_connector_id
  db_instance_connection_name = data.terraform_remote_state.foundation.outputs.db_instance_connection_name
  
  command = ["/opt/airflow/entrypoint.sh"]
  args    = ["scheduler"]

  memory = "8Gi"
  cpu    = "2"
  cpu_idle = false

  env_vars = concat(local.common_env, [
    { name = "AIRFLOW__CORE__EXECUTION_API_SERVER_URL", value = "${module.airflow_webserver.uri}/execution/" }
  ])
  secrets = local.common_secrets
}

module "airflow_dag_processor" {
  source      = "../modules/cloud-run-service"
  project_id  = var.project_id
  region      = var.region
  name        = "airflow-dag-processor-${var.environment}"
  image       = local.airflow_image
  container_port = 8080

  service_account_email       = module.airflow_iam.email
  vpc_connector_id            = data.terraform_remote_state.foundation.outputs.vpc_connector_id
  db_instance_connection_name = data.terraform_remote_state.foundation.outputs.db_instance_connection_name

  command = ["/opt/airflow/entrypoint.sh"]
  args    = ["dag-processor"]

  memory   = "2Gi"
  cpu      = "1"
  cpu_idle = false

  env_vars = concat(local.common_env, [
    { name = "AIRFLOW__CORE__EXECUTION_API_SERVER_URL", value = "${module.airflow_webserver.uri}/execution/" }
  ])
  secrets = local.common_secrets
}

module "airflow_init" {
  source      = "../modules/cloud-run-job"
  project_id  = var.project_id
  region      = var.region
  name        = "airflow-init-${var.environment}"
  image       = local.airflow_image
  
  service_account_email       = module.airflow_iam.email
  vpc_connector_id            = data.terraform_remote_state.foundation.outputs.vpc_connector_id
  db_instance_connection_name = data.terraform_remote_state.foundation.outputs.db_instance_connection_name
  
  command = ["/opt/airflow/entrypoint.sh"]
  args    = ["db-init"]
  
  env_vars = concat(local.common_env, [
    { name = "AIRFLOW_ADMIN_PASSWORD", value = var.airflow_admin_password }
  ])
  secrets  = local.common_secrets
}

output "webserver_url" {
  value = module.airflow_webserver.uri
}
