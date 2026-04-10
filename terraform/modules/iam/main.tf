# ── Service Account ──────────────────────────────────────────────
# This is the "identity" your Cloud Run services run as
# Think of it as: "I am the Airflow service, and I have permission to..."
resource "google_service_account" "airflow" {
  account_id   = "airflow-cloudrun-${var.environment}"
  display_name = "Airflow Cloud Run Service Account"
  project      = var.project_id
}

# ── Role Bindings ────────────────────────────────────────────────
# Each binding says: "this service account is allowed to do X"

# Can connect to Cloud SQL
resource "google_project_iam_member" "cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.airflow.email}"
}

# Can read secrets from Secret Manager
resource "google_project_iam_member" "secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.airflow.email}"
}

# Can read/write to GCS buckets (for DVC and data uploads)
resource "google_project_iam_member" "storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.airflow.email}"
}

# Can pull Docker images from Artifact Registry
resource "google_project_iam_member" "artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.airflow.email}"
}

# Can invoke other Cloud Run services (scheduler calls webserver)
resource "google_project_iam_member" "run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.airflow.email}"
}

# Can write logs to Cloud Logging
resource "google_project_iam_member" "log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.airflow.email}"
}
