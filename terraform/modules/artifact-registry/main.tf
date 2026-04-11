resource "google_artifact_registry_repository" "airflow" {
  location      = var.region
  project       = var.project_id
  repository_id = "airflow-images-${var.environment}"
  format        = "DOCKER"
  description   = "Docker images for Airflow deployment"
}
