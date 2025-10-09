# RevAI Infrastructure Deployment Guide

This guide provides step-by-step instructions for deploying the RevAI infrastructure across different environments.

## 🏗️ Architecture Overview

The RevAI infrastructure consists of:

- **Azure Kubernetes Service (AKS)** - Container orchestration
- **Azure Database for PostgreSQL** - Primary database
- **Azure Cache for Redis** - Caching and session storage
- **Azure Key Vault** - Secrets management
- **Azure Container Registry** - Container image storage
- **Azure Service Bus** - Message queuing
- **Azure Monitor** - Observability and monitoring

## 📋 Prerequisites

### Required Tools

- [Terraform](https://www.terraform.io/downloads.html) >= 1.9.2
- [Helm](https://helm.sh/docs/intro/install/) >= 3.12.0
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) >= 2.50.0
- [kubectl](https://kubernetes.io/docs/tasks/tools/) >= 1.28.0
- [Git](https://git-scm.com/downloads) >= 2.30.0

### Azure Requirements

- Azure subscription with appropriate permissions
- Resource group creation permissions
- AKS cluster creation permissions
- Key Vault creation permissions

### Authentication Setup

1. **Azure CLI Login**:
   ```bash
   az login
   az account set --subscription "your-subscription-id"
   ```

2. **Service Principal (for CI/CD)**:
   ```bash
   az ad sp create-for-rbac --name "revai-infra-sp" \
     --role Contributor \
     --scopes /subscriptions/your-subscription-id
   ```

3. **Environment Variables**:
   ```bash
   export ARM_CLIENT_ID="your-client-id"
   export ARM_CLIENT_SECRET="your-client-secret"
   export ARM_SUBSCRIPTION_ID="your-subscription-id"
   export ARM_TENANT_ID="your-tenant-id"
   ```

## 🚀 Deployment Process

### 1. Development Environment

#### Step 1: Clone Repository
```bash
git clone https://github.com/Crenovent/infra.git
cd infra
```

#### Step 2: Configure Development Environment
```bash
cd terraform/environments/dev
cp dev.tfvars.example dev.tfvars
# Edit dev.tfvars with your values
```

#### Step 3: Initialize Terraform
```bash
terraform init
```

#### Step 4: Plan Deployment
```bash
terraform plan -var-file="dev.tfvars"
```

#### Step 5: Apply Infrastructure
```bash
terraform apply -var-file="dev.tfvars"
```

#### Step 6: Deploy Applications
```bash
# Get AKS credentials
az aks get-credentials --resource-group rg-revai-dev --name revai-dev-aks

# Create namespace
kubectl create namespace revai

# Deploy orchestrator
helm install orchestrator ./helm/orchestrator \
  --namespace revai \
  --set secrets.postgresql_password="your-password" \
  --set secrets.redis_password="your-password" \
  --set secrets.jwt_secret="your-jwt-secret"

# Deploy evidence worker
helm install evidence-worker ./helm/evidence-worker \
  --namespace revai \
  --set secrets.postgresql_password="your-password" \
  --set secrets.redis_password="your-password" \
  --set secrets.jwt_secret="your-jwt-secret"
```

### 2. Staging Environment

#### Step 1: Configure Staging Environment
```bash
cd terraform/environments/staging
cp staging.tfvars.example staging.tfvars
# Edit staging.tfvars with your values
```

#### Step 2: Deploy Infrastructure
```bash
terraform init
terraform plan -var-file="staging.tfvars"
terraform apply -var-file="staging.tfvars"
```

#### Step 3: Deploy Applications
```bash
# Get AKS credentials
az aks get-credentials --resource-group rg-revai-staging --name revai-staging-aks

# Deploy applications
helm upgrade --install orchestrator ./helm/orchestrator \
  --namespace revai \
  --values ./helm/orchestrator/values-staging.yaml

helm upgrade --install evidence-worker ./helm/evidence-worker \
  --namespace revai \
  --values ./helm/evidence-worker/values-staging.yaml
```

### 3. Production Environment

#### Step 1: Configure Production Environment
```bash
cd terraform/environments/prod
cp prod.tfvars.example prod.tfvars
# Edit prod.tfvars with your values
```

#### Step 2: Deploy Infrastructure
```bash
terraform init
terraform plan -var-file="prod.tfvars"
terraform apply -var-file="prod.tfvars"
```

#### Step 3: Deploy Applications
```bash
# Get AKS credentials
az aks get-credentials --resource-group rg-revai-prod --name revai-prod-aks

# Deploy applications
helm upgrade --install orchestrator ./helm/orchestrator \
  --namespace revai \
  --values ./helm/orchestrator/values-prod.yaml

helm upgrade --install evidence-worker ./helm/evidence-worker \
  --namespace revai \
  --values ./helm/evidence-worker/values-prod.yaml
```

## 🔧 Configuration Management

### Environment Variables

Each environment requires specific configuration:

#### Development
- **Location**: Central India
- **Node Count**: 2
- **VM Size**: Standard_D2s_v4
- **Storage**: 32GB PostgreSQL, 1GB Redis
- **Monitoring**: Basic

#### Staging
- **Location**: West Europe
- **Node Count**: 3
- **VM Size**: Standard_D4s_v4
- **Storage**: 64GB PostgreSQL, 2GB Redis
- **Monitoring**: Enhanced

#### Production
- **Location**: Multiple regions
- **Node Count**: 5+
- **VM Size**: Standard_D8s_v4
- **Storage**: 128GB+ PostgreSQL, 4GB+ Redis
- **Monitoring**: Full observability

### Secrets Management

1. **Azure Key Vault Integration**:
   ```bash
   # Store secrets in Key Vault
   az keyvault secret set --vault-name "revai-dev-kv" \
     --name "postgresql-password" \
     --value "your-password"
   ```

2. **Helm Secrets**:
   ```bash
   # Use external secrets operator
   helm install external-secrets external-secrets/external-secrets \
     --namespace external-secrets-system \
     --create-namespace
   ```

### Network Configuration

1. **Virtual Network Setup**:
   - VNet: 10.0.0.0/16
   - AKS Subnet: 10.0.1.0/24
   - Pod Subnet: 10.0.2.0/24

2. **Security Groups**:
   - Allow inbound traffic on ports 80, 443
   - Allow outbound traffic to Azure services
   - Block unnecessary traffic

## 📊 Monitoring and Observability

### Application Insights Setup

1. **Enable Application Insights**:
   ```bash
   # Already configured in Terraform
   # Check Application Insights workspace
   az monitor app-insights component show \
     --resource-group rg-revai-dev \
     --app revai-dev-ai
   ```

2. **Configure Logging**:
   ```bash
   # Enable container insights
   az aks enable-addons \
     --resource-group rg-revai-dev \
     --name revai-dev-aks \
     --addons monitoring
   ```

### Prometheus and Grafana

1. **Install Prometheus**:
   ```bash
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm install prometheus prometheus-community/kube-prometheus-stack \
     --namespace monitoring \
     --create-namespace
   ```

2. **Install Grafana**:
   ```bash
   # Grafana is included in kube-prometheus-stack
   kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring
   ```

## 🔒 Security Configuration

### RBAC Setup

1. **Create Service Accounts**:
   ```bash
   kubectl create serviceaccount orchestrator-sa -n revai
   kubectl create serviceaccount evidence-worker-sa -n revai
   ```

2. **Assign Roles**:
   ```bash
   kubectl create rolebinding orchestrator-rolebinding \
     --clusterrole=edit \
     --serviceaccount=revai:orchestrator-sa \
     --namespace=revai
   ```

### Network Policies

1. **Enable Network Policies**:
   ```bash
   # Already enabled in AKS configuration
   # Apply network policies
   kubectl apply -f ./k8s/network-policies/
   ```

### Pod Security Standards

1. **Enable Pod Security Standards**:
   ```bash
   kubectl label namespace revai pod-security.kubernetes.io/enforce=restricted
   ```

## 🚨 Troubleshooting

### Common Issues

#### 1. Terraform State Issues
```bash
# Refresh state
terraform refresh -var-file="dev.tfvars"

# Import existing resources
terraform import azurerm_resource_group.main /subscriptions/.../resourceGroups/rg-revai-dev
```

#### 2. AKS Connection Issues
```bash
# Reset AKS credentials
az aks get-credentials --resource-group rg-revai-dev --name revai-dev-aks --overwrite-existing

# Check cluster status
az aks show --resource-group rg-revai-dev --name revai-dev-aks
```

#### 3. Helm Deployment Issues
```bash
# Check Helm releases
helm list -n revai

# Check pod status
kubectl get pods -n revai

# Check logs
kubectl logs -n revai deployment/orchestrator
```

#### 4. Database Connection Issues
```bash
# Check PostgreSQL status
az postgres flexible-server show --resource-group rg-revai-dev --name revai-dev-postgres

# Test connection
kubectl run postgresql-client --rm -i --tty --restart=Never \
  --image=postgres:15 \
  --env="PGPASSWORD=your-password" \
  -- psql -h revai-dev-postgres.postgres.database.azure.com -U postgresadmin -d revai_dev
```

### Log Analysis

1. **Application Logs**:
   ```bash
   kubectl logs -n revai deployment/orchestrator --tail=100
   kubectl logs -n revai deployment/evidence-worker --tail=100
   ```

2. **System Logs**:
   ```bash
   kubectl logs -n kube-system deployment/azure-ip-masq-agent
   kubectl logs -n kube-system deployment/cloud-node-manager
   ```

3. **Azure Monitor Logs**:
   ```bash
   # Query Application Insights
   az monitor app-insights query \
     --app revai-dev-ai \
     --analytics-query "traces | where timestamp > ago(1h)"
   ```

## 🔄 Maintenance Procedures

### Regular Maintenance

1. **Weekly Tasks**:
   - Review resource usage
   - Check security updates
   - Monitor cost trends
   - Review logs for errors

2. **Monthly Tasks**:
   - Update Terraform modules
   - Review and rotate secrets
   - Update Helm charts
   - Conduct security scans

3. **Quarterly Tasks**:
   - Review architecture
   - Update documentation
   - Conduct disaster recovery tests
   - Review compliance requirements

### Backup Procedures

1. **Database Backups**:
   ```bash
   # Automated backups are enabled
   # Manual backup if needed
   az postgres flexible-server backup create \
     --resource-group rg-revai-dev \
     --name revai-dev-postgres
   ```

2. **Configuration Backups**:
   ```bash
   # Backup Terraform state
   terraform state pull > terraform-state-backup.json

   # Backup Helm releases
   helm list -n revai -o yaml > helm-releases-backup.yaml
   ```

### Disaster Recovery

1. **Recovery Procedures**:
   - Document recovery steps
   - Test recovery procedures
   - Maintain recovery documentation
   - Train team on recovery procedures

2. **Recovery Testing**:
   - Monthly DR drills
   - Document test results
   - Update procedures based on tests
   - Maintain recovery runbooks

## 📞 Support and Escalation

### Support Channels

1. **GitHub Issues**: For bugs and feature requests
2. **Slack**: #devops channel for general questions
3. **Email**: devops@crenovent.com for urgent issues
4. **Emergency**: #devops-alerts Slack channel

### Escalation Procedures

1. **Level 1**: DevOps Team
2. **Level 2**: Engineering Manager
3. **Level 3**: CTO
4. **Level 4**: Executive Team

### Emergency Contacts

- **DevOps Lead**: devops-lead@crenovent.com
- **Engineering Manager**: eng-manager@crenovent.com
- **CTO**: cto@crenovent.com

---

## 📚 Additional Resources

- [Terraform Azure Provider Documentation](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)
- [Helm Documentation](https://helm.sh/docs/)
- [Azure Kubernetes Service Documentation](https://docs.microsoft.com/en-us/azure/aks/)
- [RevAI Architecture Documentation](./architecture.md)
- [Security Guidelines](./security.md)

For additional help, please refer to the [Contributing Guide](./contributing.md) or contact the DevOps team.
