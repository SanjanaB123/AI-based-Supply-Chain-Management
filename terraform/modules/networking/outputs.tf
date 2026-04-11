output "network_id" {
  description = "VPC network ID"
  value       = google_compute_network.airflow_vpc.id
}

output "private_vpc_connection" {
  description = "Private VPC connection for Cloud SQL"
  value       = google_service_networking_connection.private_vpc_connection
}

output "vpc_connector_id" {
  description = "VPC Access connector ID for Cloud Run"
  value       = google_vpc_access_connector.airflow_connector.id
}
