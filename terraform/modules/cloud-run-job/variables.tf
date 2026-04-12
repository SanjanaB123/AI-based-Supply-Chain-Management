variable "project_id" { type = string }
variable "region" { type = string }
variable "name" { type = string }
variable "image" { type = string }
variable "service_account_email" { type = string }
variable "vpc_connector_id" { type = string; default = null }
variable "db_instance_connection_name" { type = string; default = null }
variable "command" { type = list(string); default = [] }
variable "args" { type = list(string); default = [] }
variable "env_vars" { 
  type = list(object({ name = string, value = string }))
  default = [] 
}
variable "secrets" { 
  type = list(object({ name = string, secret_id = string }))
  default = [] 
}
