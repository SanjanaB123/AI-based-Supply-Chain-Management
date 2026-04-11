terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Store Terraform state in GCS so it's not lost
  backend "gcs" {
    bucket = "mlops-project-488302-tfstate"
    prefix = "airflow"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
