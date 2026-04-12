resource "google_cloud_run_v2_job" "default" {
  name     = var.name
  location = var.region
  project  = var.project_id

  template {
    template {
      service_account = var.service_account_email
      
      dynamic "vpc_access" {
        for_each = var.vpc_connector_id != null && var.vpc_connector_id != "" ? [1] : []
        content {
          connector = var.vpc_connector_id
          egress    = "ALL_TRAFFIC"
        }
      }

      containers {
        image = var.image
        
        command = var.command
        args    = var.args

        dynamic "env" {
          for_each = var.env_vars
          content {
            name  = env.value["name"]
            value = env.value["value"]
          }
        }

        dynamic "env" {
          for_each = var.secrets
          content {
            name = env.value["name"]
            value_source {
              secret_key_ref {
                secret  = env.value["secret_id"]
                version = "latest"
              }
            }
          }
        }

        dynamic "volume_mounts" {
          for_each = var.db_instance_connection_name != null && var.db_instance_connection_name != "" ? [1] : []
          content {
            name       = "cloudsql"
            mount_path = "/cloudsql"
          }
        }
      }

      dynamic "volumes" {
        for_each = var.db_instance_connection_name != null && var.db_instance_connection_name != "" ? [1] : []
        content {
          name = "cloudsql"
          cloud_sql_instance {
            instances = [var.db_instance_connection_name]
          }
        }
      }
    }
  }
}
