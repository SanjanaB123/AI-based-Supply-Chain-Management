output "service_account_email" {
  description = "Airflow service account email"
  value       = google_service_account.airflow.email
}

output "service_account_id" {
  description = "Airflow service account ID"
  value       = google_service_account.airflow.id
}
