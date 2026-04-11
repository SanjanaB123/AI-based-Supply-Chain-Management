output "repository_url" {
  description = "Full URL of the Artifact Registry repository"
  value       = "${google_artifact_registry_repository.airflow.location}-docker.pkg.dev/${google_artifact_registry_repository.airflow.project}/${google_artifact_registry_repository.airflow.repository_id}"
}
