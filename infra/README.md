# RevAI Infrastructure Repository

This repository contains the infrastructure-as-code (IaC) components for the RevAI platform, including Terraform modules for Azure resources and Helm charts for Kubernetes deployments.

## 🏗️ Architecture Overview

The infrastructure supports a multi-region, multi-tenant SaaS platform with the following components:

- **Kubernetes Clusters** (AKS) across multiple regions
- **Storage** (Azure Storage Accounts, Azure Database for PostgreSQL)
- **Message Bus** (Azure Service Bus)
- **Secrets Management** (Azure Key Vault)
- **Container Registry** (Azure Container Registry)
- **Monitoring & Observability** (Azure Monitor, Application Insights)

## 📁 Repository Structure

```
infra/
├── terraform/                 # Terraform modules and configurations
│   ├── modules/              # Reusable Terraform modules
│   │   ├── aks/              # Azure Kubernetes Service module
│   │   ├── storage/          # Storage accounts and databases
│   │   ├── networking/       # VNet, subnets, security groups
│   │   ├── security/         # Key Vault, RBAC, policies
│   │   └── monitoring/       # Log Analytics, Application Insights
│   ├── environments/          # Environment-specific configurations
│   │   ├── dev/              # Development environment
│   │   ├── staging/          # Staging environment
│   │   └── prod/             # Production environment
│   └── shared/               # Shared resources (DNS, etc.)
├── helm/                     # Helm charts for application deployment
│   ├── orchestrator/         # Orchestrator service chart
│   ├── evidence-worker/      # Evidence worker service chart
│   ├── ai-service/          # AI service chart
│   ├── backend-api/         # Backend API chart
│   └── frontend/            # Frontend application chart
├── scripts/                 # Deployment and utility scripts
├── docs/                    # Documentation
└── .github/                 # GitHub Actions workflows
```

## 🚀 Quick Start

### Prerequisites

- [Terraform](https://www.terraform.io/downloads.html) >= 1.9.2
- [Helm](https://helm.sh/docs/intro/install/) >= 3.12.0
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) >= 2.50.0
- [kubectl](https://kubernetes.io/docs/tasks/tools/) >= 1.28.0

### Authentication

1. **Azure Authentication**:
   ```bash
   az login
   az account set --subscription "your-subscription-id"
   ```

2. **Terraform Backend** (for state management):
   ```bash
   export ARM_CLIENT_ID="your-client-id"
   export ARM_CLIENT_SECRET="your-client-secret"
   export ARM_SUBSCRIPTION_ID="your-subscription-id"
   export ARM_TENANT_ID="your-tenant-id"
   ```

### Deploy Infrastructure

1. **Initialize Terraform**:
   ```bash
   cd terraform/environments/dev
   terraform init
   ```

2. **Plan Deployment**:
   ```bash
   terraform plan -var-file="dev.tfvars"
   ```

3. **Apply Infrastructure**:
   ```bash
   terraform apply -var-file="dev.tfvars"
   ```

### Deploy Applications

1. **Add Helm Repository**:
   ```bash
   helm repo add revai ./helm
   helm repo update
   ```

2. **Deploy Services**:
   ```bash
   helm install orchestrator revai/orchestrator -n revai
   helm install evidence-worker revai/evidence-worker -n revai
   ```

## 🔧 Development Workflow

### Making Changes

1. **Create Feature Branch**:
   ```bash
   git checkout -b feature/infra-update
   ```

2. **Make Changes**:
   - Update Terraform modules
   - Modify Helm charts
   - Update documentation

3. **Test Changes**:
   ```bash
   # Terraform validation
   terraform validate
   terraform plan
   
   # Helm validation
   helm lint ./helm/orchestrator
   ```

4. **Create Pull Request**:
   - Ensure CI passes
   - Request review from DevOps team
   - Merge after approval

### Branch Protection Rules

- **main**: Production-ready code only
- **develop**: Integration branch for features
- **feature/***: Feature development branches
- **hotfix/***: Critical production fixes

## 🛡️ Security & Compliance

### Security Scanning

- **Terraform**: `tfsec`, `tflint`, `checkov`
- **Helm**: `helm lint`, `kube-score`
- **Container**: `trivy`, `grype`

### Compliance

- **SOC2 Type II** compliance ready
- **ISO 27001** security controls
- **GDPR** data protection measures
- **Azure Security Center** integration

## 📊 Monitoring & Observability

### Infrastructure Monitoring

- **Azure Monitor** for infrastructure metrics
- **Application Insights** for application telemetry
- **Log Analytics** for centralized logging
- **Azure Security Center** for security monitoring

### Cost Management

- **Azure Cost Management** integration
- **Resource tagging** for cost attribution
- **Budget alerts** and cost optimization

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

- **terraform-plan.yml**: Plan infrastructure changes
- **terraform-apply.yml**: Apply approved changes
- **helm-lint.yml**: Validate Helm charts
- **security-scan.yml**: Security scanning
- **cost-analysis.yml**: Cost impact analysis

### Deployment Gates

1. **Terraform Plan Review**: All infrastructure changes require review
2. **Security Scan**: Must pass security scanning
3. **Cost Analysis**: Cost impact must be within budget
4. **Manual Approval**: Production deployments require manual approval

## 📚 Documentation

- [Infrastructure Architecture](./docs/architecture.md)
- [Deployment Guide](./docs/deployment.md)
- [Security Guidelines](./docs/security.md)
- [Troubleshooting](./docs/troubleshooting.md)
- [Contributing](./docs/contributing.md)

## 🆘 Support

- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Slack**: #devops channel for urgent issues
- **Email**: devops@crenovent.com for security concerns

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⚠️ Important**: Never commit secrets or sensitive information to this repository. Use Azure Key Vault or GitHub Secrets for sensitive data.
