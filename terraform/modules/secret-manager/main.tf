# ── Helper: create a secret + its value ──────────────────────────
# Each secret needs TWO resources:
#   1. The secret itself (like creating an empty box)
#   2. The secret version (putting the value inside the box)

# -- Database connection string --
resource "google_secret_manager_secret" "db_connection_string" {
  secret_id = "airflow-db-connection-string-${var.environment}"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_connection_string" {
  secret      = google_secret_manager_secret.db_connection_string.id
  secret_data = var.db_connection_string
}

# -- Database password --
resource "google_secret_manager_secret" "db_password" {
  secret_id = "airflow-db-password-${var.environment}"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}

# -- Fernet key --
resource "google_secret_manager_secret" "fernet_key" {
  secret_id = "airflow-fernet-key-${var.environment}"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "fernet_key" {
  secret      = google_secret_manager_secret.fernet_key.id
  secret_data = var.airflow_fernet_key
}

# -- JWT secret --
resource "google_secret_manager_secret" "jwt_secret" {
  secret_id = "airflow-jwt-secret-${var.environment}"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "jwt_secret" {
  secret      = google_secret_manager_secret.jwt_secret.id
  secret_data = var.airflow_jwt_secret
}

# -- MongoDB URI --
resource "google_secret_manager_secret" "mongo_uri" {
  secret_id = "airflow-mongo-uri-${var.environment}"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "mongo_uri" {
  secret      = google_secret_manager_secret.mongo_uri.id
  secret_data = var.mongo_uri
}

# -- GitHub token --
resource "google_secret_manager_secret" "github_token" {
  secret_id = "airflow-github-token-${var.environment}"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "github_token" {
  secret      = google_secret_manager_secret.github_token.id
  secret_data = var.github_token
}

# -- Airflow admin password --
resource "google_secret_manager_secret" "airflow_admin_password" {
  secret_id = "airflow-admin-password-${var.environment}"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "airflow_admin_password" {
  secret      = google_secret_manager_secret.airflow_admin_password.id
  secret_data = var.airflow_admin_password
}
