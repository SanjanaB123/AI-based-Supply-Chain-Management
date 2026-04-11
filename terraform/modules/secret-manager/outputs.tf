output "db_connection_string_secret_id" {
  description = "Secret ID for DB connection string"
  value       = google_secret_manager_secret.db_connection_string.secret_id
}

output "db_password_secret_id" {
  description = "Secret ID for DB password"
  value       = google_secret_manager_secret.db_password.secret_id
}

output "fernet_key_secret_id" {
  description = "Secret ID for Fernet key"
  value       = google_secret_manager_secret.fernet_key.secret_id
}

output "jwt_secret_secret_id" {
  description = "Secret ID for JWT secret"
  value       = google_secret_manager_secret.jwt_secret.secret_id
}

output "mongo_uri_secret_id" {
  description = "Secret ID for MongoDB URI"
  value       = google_secret_manager_secret.mongo_uri.secret_id
}

output "github_token_secret_id" {
  description = "Secret ID for GitHub token"
  value       = google_secret_manager_secret.github_token.secret_id
}

output "airflow_admin_password_secret_id" {
  description = "Secret ID for Airflow admin password"
  value       = google_secret_manager_secret.airflow_admin_password.secret_id
}
