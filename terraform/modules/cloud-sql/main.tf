# Cloud SQL PostgreSQL Instance 
# This replaces the Postgres Docker container from docker-compose
resource "google_sql_database_instance" "airflow" {
  name             = "airflow-db-${var.environment}"
  database_version = "POSTGRES_15"
  region           = var.region
  project          = var.project_id

  # Wait for private VPC connection before creating
  depends_on = [var.private_vpc_connection]

  settings {
    tier              = var.db_tier
    availability_type = "ZONAL"
    disk_size         = 10
    disk_type         = "PD_SSD"

    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = var.network_id
      enable_private_path_for_google_cloud_services = true
    }

    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }
  }

  deletion_protection = false
}

# Database 
resource "google_sql_database" "airflow" {
  name     = "airflow"
  instance = google_sql_database_instance.airflow.name
  project  = var.project_id
}

# Database User 
resource "google_sql_user" "airflow" {
  name     = "airflow"
  instance = google_sql_database_instance.airflow.name
  password = var.db_password
  project  = var.project_id
}
