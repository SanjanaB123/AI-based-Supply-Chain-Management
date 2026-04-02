# ======================================================================
# CLOUD RUN MODULE
# Creates 4 Airflow services + 1 init job
#
# Think of it like this:
#   docker-compose.yaml had 5 services → we create 5 things here
#   But instead of running on your laptop, they run on GCP
# ======================================================================

# ── Local values (reused across all services) ────────────────────
locals {
  # Environment variables shared by ALL Airflow services
  # Same as the "environment" block in your docker-compose.yaml
  common_env = [
    { name = "AIRFLOW__CORE__EXECUTOR",                    value = "LocalExecutor" },
    { name = "AIRFLOW__CORE__AUTH_MANAGER",                value = "airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager" },
    { name = "AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION", value = "true" },
    { name = "AIRFLOW__CORE__LOAD_EXAMPLES",               value = "false" },
    { name = "AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK",    value = "true" },
    { name = "PYTHONPATH",                                  value = "/opt/airflow" },
    { name = "PARAMS_PATH",                                 value = "/opt/airflow/params.yaml" },
    { name = "GCS_BUCKET_NAME",                             value = var.gcs_bucket_name },
    { name = "GITHUB_REPO",                                 value = var.github_repo },
  ]

  # Secrets shared by ALL services (read from Secret Manager)
  common_secrets = [
    {
      name      = "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"
      secret_id = var.db_connection_string_secret_id
    },
    {
      name      = "AIRFLOW__CORE__FERNET_KEY"
      secret_id = var.fernet_key_secret_id
    },
    {
      name      = "AIRFLOW__API_AUTH__JWT_SECRET"
      secret_id = var.jwt_secret_secret_id
    },
    {
      name      = "MONGO_URI"
      secret_id = var.mongo_uri_secret_id
    },
    {
      name      = "GITHUB_TOKEN"
      secret_id = var.github_token_secret_id
    },
  ]
}


# ======================================================================
# SERVICE 1: WEBSERVER (api-server)
# The Airflow UI — the only service that's publicly accessible
# Same as "airflow-apiserver" in your docker-compose
# ======================================================================
resource "google_cloud_run_v2_service" "webserver" {
  name     = "airflow-webserver-${var.environment}"
  location = var.region
  project  = var.project_id

  template {
    service_account = var.service_account_email

    scaling {
      min_instance_count = 1
      max_instance_count = 2
    }

    vpc_access {
      connector = var.vpc_connector_id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [var.db_instance_connection_name]
      }
    }

    containers {
      image   = var.airflow_image
      command = ["/opt/airflow/entrypoint.sh"]
      args    = ["api-server", "--port", "8080"]


      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "4Gi"
        }
        cpu_idle = false
      }

      # Plain environment variables
      dynamic "env" {
        for_each = local.common_env
        content {
          name  = env.value.name
          value = env.value.value
        }
      }

      # The webserver points to itself for the execution API
      env {
        name  = "AIRFLOW__CORE__EXECUTION_API_SERVER_URL"
        value = "http://localhost:8080/execution/"
      }

      env {
        name  = "AIRFLOW__WEBSERVER__BASE_URL"
        value = "https://airflow-webserver-${var.environment}-${var.project_id}.${var.region}.run.app"
      }

      env {
        name  = "AIRFLOW__WEBSERVER__WORKERS"
        value = "1"
      }

      # Secret environment variables (read from Secret Manager)
      dynamic "env" {
        for_each = local.common_secrets
        content {
          name = env.value.name
          value_source {
            secret_key_ref {
              secret  = env.value.secret_id
              version = "latest"
            }
          }
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }

      startup_probe {
        http_get {
          path = "/api/v2/version"
          port = 8080
        }
        initial_delay_seconds = 90
        period_seconds        = 15
        failure_threshold     = 30
      }
    }
  }

  depends_on = [google_cloud_run_v2_service.scheduler]
}

# Make webserver publicly accessible (so you can open the Airflow UI)
resource "google_cloud_run_v2_service_iam_member" "webserver_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.webserver.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}


# ======================================================================
# SERVICE 2: SCHEDULER
# The brain — decides when to run your DAG tasks
# Same as "airflow-scheduler" in your docker-compose
# ======================================================================
resource "google_cloud_run_v2_service" "scheduler" {
  name     = "airflow-scheduler-${var.environment}"
  location = var.region
  project  = var.project_id

  template {
    service_account = var.service_account_email

    scaling {
      min_instance_count = 1
      max_instance_count = 1
    }

    vpc_access {
      connector = var.vpc_connector_id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [var.db_instance_connection_name]
      }
    }

    containers {
      image   = var.airflow_image
      command = ["/opt/airflow/entrypoint.sh"]
      args    = ["scheduler"]


      # Scheduler needs a port for health checks even though it's not HTTP
      ports {
        container_port = 8974
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
        # CRITICAL: keeps CPU always running (scheduler is a background process)
        cpu_idle = false
      }

      dynamic "env" {
        for_each = local.common_env
        content {
          name  = env.value.name
          value = env.value.value
        }
      }

      # Scheduler talks to the webserver's execution API
      env {
        name  = "AIRFLOW__CORE__EXECUTION_API_SERVER_URL"
        value = "https://airflow-webserver-${var.environment}-${var.project_id}.${var.region}.run.app/execution/"
      }

      dynamic "env" {
        for_each = local.common_secrets
        content {
          name = env.value.name
          value_source {
            secret_key_ref {
              secret  = env.value.secret_id
              version = "latest"
            }
          }
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8974
        }
        initial_delay_seconds = 60
        period_seconds        = 10
        failure_threshold     = 20
      }
    }
  }
}


# ======================================================================
# SERVICE 3: DAG PROCESSOR
# Reads your DAG Python files and understands the task dependencies
# Same as "airflow-dag-processor" in your docker-compose
# ======================================================================
resource "google_cloud_run_v2_service" "dag_processor" {
  name     = "airflow-dag-processor-${var.environment}"
  location = var.region
  project  = var.project_id

  template {
    service_account = var.service_account_email

    scaling {
      min_instance_count = 1
      max_instance_count = 1
    }

    vpc_access {
      connector = var.vpc_connector_id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [var.db_instance_connection_name]
      }
    }

    containers {
      image   = var.airflow_image
      command = ["/opt/airflow/entrypoint.sh"]
      args    = ["dag-processor"]



      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle = false
      }

      dynamic "env" {
        for_each = local.common_env
        content {
          name  = env.value.name
          value = env.value.value
        }
      }

      env {
        name  = "AIRFLOW__CORE__EXECUTION_API_SERVER_URL"
        value = "https://airflow-webserver-${var.environment}-${var.project_id}.${var.region}.run.app/execution/"
      }

      dynamic "env" {
        for_each = local.common_secrets
        content {
          name = env.value.name
          value_source {
            secret_key_ref {
              secret  = env.value.secret_id
              version = "latest"
            }
          }
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }
  }
}


# ======================================================================
# SERVICE 4: TRIGGERER
# Handles deferred/async tasks
# Same as "airflow-triggerer" in your docker-compose
# ======================================================================
resource "google_cloud_run_v2_service" "triggerer" {
  name     = "airflow-triggerer-${var.environment}"
  location = var.region
  project  = var.project_id

  template {
    service_account = var.service_account_email

    scaling {
      min_instance_count = 1
      max_instance_count = 1
    }

    vpc_access {
      connector = var.vpc_connector_id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [var.db_instance_connection_name]
      }
    }

    containers {
      image   = var.airflow_image
      command = ["/opt/airflow/entrypoint.sh"]
      args    = ["triggerer"]

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle = false
      }

      dynamic "env" {
        for_each = local.common_env
        content {
          name  = env.value.name
          value = env.value.value
        }
      }

      env {
        name  = "AIRFLOW__CORE__EXECUTION_API_SERVER_URL"
        value = "https://airflow-webserver-${var.environment}-${var.project_id}.${var.region}.run.app/execution/"
      }

      dynamic "env" {
        for_each = local.common_secrets
        content {
          name = env.value.name
          value_source {
            secret_key_ref {
              secret  = env.value.secret_id
              version = "latest"
            }
          }
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }
  }
}


# ======================================================================
# JOB: AIRFLOW INIT
# One-shot job: migrates the database and creates the admin user
# Same as "airflow-init" in your docker-compose
# Runs once during deployment, not continuously
# ======================================================================
resource "google_cloud_run_v2_job" "airflow_init" {
  name     = "airflow-init-${var.environment}"
  location = var.region
  project  = var.project_id

  template {
    template {
      service_account = var.service_account_email

      vpc_access {
        connector = var.vpc_connector_id
        egress    = "PRIVATE_RANGES_ONLY"
      }

      volumes {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [var.db_instance_connection_name]
        }
      }

      containers {
        image = var.airflow_image

        command = ["/opt/airflow/entrypoint.sh"]
        args    = ["db-init"]


        resources {
          limits = {
            cpu    = "1"
            memory = "2Gi"
          }
        }

        dynamic "env" {
          for_each = local.common_env
          content {
            name  = env.value.name
            value = env.value.value
          }
        }

        dynamic "env" {
          for_each = local.common_secrets
          content {
            name = env.value.name
            value_source {
              secret_key_ref {
                secret  = env.value.secret_id
                version = "latest"
              }
            }
          }
        }

        volume_mounts {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }
      }

      max_retries = 1
      timeout     = "600s"
    }
  }
}
