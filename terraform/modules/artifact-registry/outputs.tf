output "repository_id" {
  description = "Artifact Registry repository ID"
  value       = google_artifact_registry_repository.airflow.repository_id
}

output "repository_url" {
  description = "Full Docker image URL prefix"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.airflow.repository_id}"
}
