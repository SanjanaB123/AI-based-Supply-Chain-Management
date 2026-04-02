# VPC Network 
# A private network so Cloud Run can talk to Cloud SQL securely
resource "google_compute_network" "airflow_vpc" {
  name                    = "airflow-vpc-${var.environment}"
  auto_create_subnetworks = false
  project                 = var.project_id
}

# Subnet
resource "google_compute_subnetwork" "airflow_subnet" {
  name          = "airflow-subnet-${var.environment}"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.airflow_vpc.id
  project       = var.project_id
}

# Private IP range for Cloud SQL
# Reserves an IP range so Cloud SQL gets a private IP (not public)
resource "google_compute_global_address" "private_ip_range" {
  name          = "airflow-private-ip-${var.environment}"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.airflow_vpc.id
  project       = var.project_id
}

# Private Services Connection 
# Connects Google's internal network to your VPC (needed for Cloud SQL private IP)
resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.airflow_vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_range.name]
}

# VPC Connector 
# The "bridge" that lets Cloud Run (serverless) reach into your VPC
resource "google_vpc_access_connector" "airflow_connector" {
  name          = "airflow-vpc-conn-${var.environment}"
  region        = var.region
  project       = var.project_id
  ip_cidr_range = "10.8.0.0/28"
  network       = google_compute_network.airflow_vpc.name

  min_instances = 2
  max_instances = 3
}
