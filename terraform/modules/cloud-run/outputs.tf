output "webserver_url" {
  description = "Airflow webserver URL (your Airflow UI)"
  value       = google_cloud_run_v2_service.webserver.uri
}

output "scheduler_url" {
  description = "Scheduler service URL (internal)"
  value       = google_cloud_run_v2_service.scheduler.uri
}

output "init_job_name" {
  description = "Init job name (for manual execution)"
  value       = google_cloud_run_v2_job.airflow_init.name
}
