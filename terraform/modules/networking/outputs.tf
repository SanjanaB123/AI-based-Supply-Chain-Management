output "network_id" {
  description = "VPC network ID"
  value       = google_compute_network.airflow_vpc.id
}

output "network_name" {
  description = "VPC network name"
  value       = google_compute_network.airflow_vpc.name
}

output "subnet_id" {
  description = "Subnet ID"
  value       = google_compute_subnetwork.airflow_subnet.id
}

output "vpc_connector_id" {
  description = "VPC connector ID for Cloud Run"
  value       = google_vpc_access_connector.airflow_connector.id
}

output "private_vpc_connection" {
  description = "Private VPC connection (Cloud SQL depends on this)"
  value       = google_service_networking_connection.private_vpc_connection
}
