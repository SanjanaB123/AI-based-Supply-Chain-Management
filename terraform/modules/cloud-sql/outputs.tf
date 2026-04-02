output "instance_name" {
  description = "Cloud SQL instance name"
  value       = google_sql_database_instance.airflow.name
}

output "instance_connection_name" {
  description = "Cloud SQL connection name (project:region:instance)"
  value       = google_sql_database_instance.airflow.connection_name
}

output "private_ip" {
  description = "Cloud SQL private IP address"
  value       = google_sql_database_instance.airflow.private_ip_address
}

output "database_name" {
  description = "Database name"
  value       = google_sql_database.airflow.name
}
