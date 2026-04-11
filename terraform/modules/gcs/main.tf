# ======================================================================
# GCS MODULE
# Creates the GCS bucket used by the data pipeline for DVC storage,
# feature uploads, and model artifacts
# ======================================================================

resource "google_storage_bucket" "pipeline" {
  name     = "${var.gcs_bucket_name}-${var.environment}"
  location = var.region
  project  = var.project_id

  # Prevent accidental deletion of pipeline data
  force_destroy = var.environment == "prod" ? false : true

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90 # days
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
}
