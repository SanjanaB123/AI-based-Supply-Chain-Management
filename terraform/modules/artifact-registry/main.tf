# ── Artifact Registry ────────────────────────────────────────────
# This is where your custom Airflow Docker image gets stored
# Think of it as a private Docker Hub, but on GCP
resource "google_artifact_registry_repository" "airflow" {
  location      = var.region
  repository_id = "airflow-images-${var.environment}"
  format        = "DOCKER"
  project       = var.project_id

  description = "Docker images for Airflow Cloud Run deployment"
}
