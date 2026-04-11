output "service_account_email" {
  description = "Email of the Airflow service account"
  value       = google_service_account.airflow.email
}
