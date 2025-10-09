# ================================================================================
# AKS Module Variables
# ================================================================================

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Network Configuration
variable "vnet_address_space" {
  description = "Address space for the virtual network"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "subnet_address_prefixes" {
  description = "Address prefixes for the AKS subnet"
  type        = list(string)
  default     = ["10.0.1.0/24"]
}

variable "pod_subnet_address_prefixes" {
  description = "Address prefixes for the pod subnet"
  type        = list(string)
  default     = ["10.0.2.0/24"]
}

# AKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_count" {
  description = "Number of nodes in the default node pool"
  type        = number
  default     = 3
}

variable "vm_size" {
  description = "VM size for the default node pool"
  type        = string
  default     = "Standard_D4s_v4"
}

variable "enable_auto_scaling" {
  description = "Enable auto scaling for the default node pool"
  type        = bool
  default     = true
}

variable "min_count" {
  description = "Minimum number of nodes"
  type        = number
  default     = 2
}

variable "max_count" {
  description = "Maximum number of nodes"
  type        = number
  default     = 10
}

variable "enable_monitoring" {
  description = "Enable monitoring"
  type        = bool
  default     = true
}

variable "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID for monitoring"
  type        = string
  default     = ""
}

variable "enable_nsg" {
  description = "Enable Network Security Group"
  type        = bool
  default     = true
}

# Application Node Pool Configuration
variable "enable_app_node_pool" {
  description = "Enable application node pool"
  type        = bool
  default     = true
}

variable "app_vm_size" {
  description = "VM size for the application node pool"
  type        = string
  default     = "Standard_D8s_v4"
}

variable "app_node_count" {
  description = "Number of nodes in the application node pool"
  type        = number
  default     = 2
}

variable "app_min_count" {
  description = "Minimum number of nodes in the application node pool"
  type        = number
  default     = 1
}

variable "app_max_count" {
  description = "Maximum number of nodes in the application node pool"
  type        = number
  default     = 20
}
