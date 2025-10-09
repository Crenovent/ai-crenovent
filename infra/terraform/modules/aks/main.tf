# ================================================================================
# AKS Module - Azure Kubernetes Service
# ================================================================================

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 4.0.0"
    }
  }
}

# Resource Group
resource "azurerm_resource_group" "aks" {
  name     = "${var.name_prefix}-aks-rg"
  location = var.location
  tags     = var.tags
}

# Virtual Network
resource "azurerm_virtual_network" "aks" {
  name                = "${var.name_prefix}-aks-vnet"
  location            = azurerm_resource_group.aks.location
  resource_group_name = azurerm_resource_group.aks.name
  address_space       = var.vnet_address_space
  tags                = var.tags
}

# Subnets
resource "azurerm_subnet" "aks" {
  name                 = "${var.name_prefix}-aks-subnet"
  resource_group_name  = azurerm_resource_group.aks.name
  virtual_network_name = azurerm_virtual_network.aks.name
  address_prefixes     = var.subnet_address_prefixes
}

resource "azurerm_subnet" "aks_pods" {
  name                 = "${var.name_prefix}-aks-pods-subnet"
  resource_group_name  = azurerm_resource_group.aks.name
  virtual_network_name = azurerm_virtual_network.aks.name
  address_prefixes     = var.pod_subnet_address_prefixes
}

# Network Security Group
resource "azurerm_network_security_group" "aks" {
  count               = var.enable_nsg ? 1 : 0
  name                = "${var.name_prefix}-aks-nsg"
  location            = azurerm_resource_group.aks.location
  resource_group_name = azurerm_resource_group.aks.name
  tags                = var.tags
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "aks" {
  name                = "${var.name_prefix}-aks"
  location            = azurerm_resource_group.aks.location
  resource_group_name = azurerm_resource_group.aks.name
  dns_prefix          = "${var.name_prefix}-aks"
  kubernetes_version  = var.kubernetes_version
  tags                = var.tags

  default_node_pool {
    name                = "system"
    node_count         = var.node_count
    vm_size            = var.vm_size
    vnet_subnet_id     = azurerm_subnet.aks.id
    enable_auto_scaling = var.enable_auto_scaling
    min_count          = var.min_count
    max_count          = var.max_count
    os_disk_size_gb    = 50
    type               = "VirtualMachineScaleSets"
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    network_policy    = "azure"
    service_cidr      = "10.1.0.0/16"
    dns_service_ip   = "10.1.0.10"
    pod_cidr         = var.pod_subnet_address_prefixes[0]
  }

  # Enable monitoring
  dynamic "oms_agent" {
    for_each = var.enable_monitoring ? [1] : []
    content {
      log_analytics_workspace_id = var.log_analytics_workspace_id
    }
  }

  # Enable RBAC
  role_based_access_control_enabled = true

  # Enable Azure Policy
  azure_policy_enabled = true

  # Enable workload identity
  workload_identity_enabled = true
  oidc_issuer_enabled       = true

  depends_on = [
    azurerm_subnet.aks,
    azurerm_subnet.aks_pods
  ]
}

# Additional Node Pool for Applications
resource "azurerm_kubernetes_cluster_node_pool" "app" {
  count                 = var.enable_app_node_pool ? 1 : 0
  name                  = "app"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size              = var.app_vm_size
  node_count           = var.app_node_count
  enable_auto_scaling   = true
  min_count            = var.app_min_count
  max_count            = var.app_max_count
  os_disk_size_gb      = 100
  vnet_subnet_id       = azurerm_subnet.aks.id
  node_taints          = ["workload=app:NoSchedule"]
  tags                 = var.tags
}

# Outputs
output "aks_cluster_id" {
  description = "The ID of the AKS cluster"
  value       = azurerm_kubernetes_cluster.aks.id
}

output "aks_cluster_name" {
  description = "The name of the AKS cluster"
  value       = azurerm_kubernetes_cluster.aks.name
}

output "aks_cluster_fqdn" {
  description = "The FQDN of the AKS cluster"
  value       = azurerm_kubernetes_cluster.aks.fqdn
}

output "aks_cluster_kube_config" {
  description = "The kubeconfig for the AKS cluster"
  value       = azurerm_kubernetes_cluster.aks.kube_config_raw
  sensitive   = true
}

output "aks_cluster_identity" {
  description = "The managed identity of the AKS cluster"
  value       = azurerm_kubernetes_cluster.aks.identity
}

output "resource_group_name" {
  description = "The name of the resource group"
  value       = azurerm_resource_group.aks.name
}

output "vnet_id" {
  description = "The ID of the virtual network"
  value       = azurerm_virtual_network.aks.id
}

output "subnet_id" {
  description = "The ID of the AKS subnet"
  value       = azurerm_subnet.aks.id
}
