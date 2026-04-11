output "airflow_webserver_url" {
  description = "URL to access the Airflow UI"
  value       = module.cloud_run.webserver_url
}

output "cloud_sql_instance" {
  description = "Cloud SQL instance connection name"
  value       = module.cloud_sql.instance_connection_name
}

output "artifact_registry_url" {
  description = "Docker image repository URL"
  value       = module.artifact_registry.repository_url
}

output "init_job_name" {
  description = "Run this to initialize the database"
  value       = "gcloud run jobs execute ${module.cloud_run.init_job_name} --region ${var.region}"
}
