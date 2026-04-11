output "private_ip" {
  description = "Private IP address of the Cloud SQL instance"
  value       = google_sql_database_instance.airflow.private_ip_address
}

output "instance_connection_name" {
  description = "Cloud SQL instance connection name (project:region:instance)"
  value       = google_sql_database_instance.airflow.connection_name
}
