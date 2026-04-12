resource "google_service_account" "sa" {
  account_id   = var.account_id
  display_name = var.display_name
  project      = var.project_id
}

resource "google_project_iam_member" "role_binding" {
  for_each = toset(var.roles)
  project  = var.project_id
  role     = each.key
  member   = "serviceAccount:${google_service_account.sa.email}"
}
