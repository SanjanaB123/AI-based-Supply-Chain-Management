resource "google_service_account" "airflow" {
  account_id   = "airflow-sa-${var.environment}"
  display_name = "Airflow Cloud Run Service Account (${var.environment})"
  project      = var.project_id
}

locals {
  sa_roles = [
    "roles/cloudsql.client",
    "roles/secretmanager.secretAccessor",
    "roles/storage.objectAdmin",
    "roles/artifactregistry.reader",
    "roles/run.invoker",
    "roles/logging.logWriter",
  ]
}

resource "google_project_iam_member" "airflow_roles" {
  for_each = toset(local.sa_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.airflow.email}"
}
