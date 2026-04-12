output "vpc_connector_id" {
  value = module.networking.vpc_connector_id
}

output "db_instance_connection_name" {
  value = module.cloud_sql.instance_connection_name
}

output "artifact_repository_url" {
  value = module.artifact_registry.repository_url
}

output "db_connection_string_secret_id" {
  value = module.secret_manager.db_connection_string_secret_id
}

output "fernet_key_secret_id" {
  value = module.secret_manager.fernet_key_secret_id
}

output "jwt_secret_secret_id" {
  value = module.secret_manager.jwt_secret_secret_id
}

output "mongo_uri_secret_id" {
  value = module.secret_manager.mongo_uri_secret_id
}

output "github_token_secret_id" {
  value = module.secret_manager.github_token_secret_id
}
