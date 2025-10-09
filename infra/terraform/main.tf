terraform {
  required_version = ">= 1.9.2"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 4.0.0, < 5.0.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    http = {
      source  = "hashicorp/http"
      version = "~> 3.4"
    }
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    key_vault {
      purge_soft_delete_on_destroy = true
    }
    log_analytics_workspace {
      permanently_delete_on_destroy = true
    }
  }
}

# Data sources
data "azurerm_subscription" "current" {}
data "azurerm_client_config" "current" {}

# Random resources for unique naming
resource "random_integer" "suffix" {
  min = 100
  max = 999
}

resource "random_pet" "name" {
  length    = 2
  separator = "-"
}

locals {
  # Naming convention: {environment}-{service}-{region}-{suffix}
  name_prefix = "${var.environment}-${random_pet.name.id}-${random_integer.suffix.result}"
  
  # Common tags
  common_tags = {
    Environment   = var.environment
    Project       = "RevAI"
    ManagedBy     = "Terraform"
    Owner         = var.owner
    CostCenter    = var.cost_center
    Compliance    = var.compliance_level
  }
}
