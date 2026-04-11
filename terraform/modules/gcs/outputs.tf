output "bucket_name" {
  description = "Name of the GCS bucket"
  value       = google_storage_bucket.pipeline.name
}

output "bucket_url" {
  description = "URL of the GCS bucket"
  value       = google_storage_bucket.pipeline.url
}
