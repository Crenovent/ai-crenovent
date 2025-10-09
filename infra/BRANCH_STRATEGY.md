# Infrastructure Branch Strategy

## Overview
This document outlines the **production-focused** deployment strategy for RevAI infrastructure. Development and UAT environments are currently **disabled** to focus on production deployment.

## Current Branch-to-Environment Mapping

| Branch | Environment | Status | Deployment Trigger | Approval Required |
|--------|-------------|--------|-------------------|-------------------|
| ~~`dev`~~ | ~~Development~~ | **DISABLED** | ~~Auto on push~~ | ~~No~~ |
| ~~`uat`~~ | ~~Staging/UAT~~ | **DISABLED** | ~~Auto on push~~ | ~~No~~ |
| `main` | Production | **ACTIVE** | Auto on push | **Yes** |

## Current Deployment Flow

### Production Environment (`main` branch) - **ACTIVE**
- **Trigger**: Push to `main` branch
- **Environment**: Production
- **Region**: East US
- **Resources**: Premium SKUs for production
- **Approval**: **Manual approval required**
- **Purpose**: Live production environment

### Development Environment (`dev` branch) - **DISABLED**
- **Status**: Temporarily disabled
- **Reason**: Focus on production deployment
- **Re-enable**: Uncomment dev configurations in workflow

### UAT Environment (`uat` branch) - **DISABLED**
- **Status**: Temporarily disabled
- **Reason**: Focus on production deployment
- **Re-enable**: Uncomment uat configurations in workflow

## Workflow Features

### Security & Compliance
- **Terraform Security Scanning**: tfsec and Checkov analysis
- **Helm Chart Validation**: Linting and templating tests
- **Cost Analysis**: Resource impact assessment on PRs
- **Manual Approval**: Required for production deployments

### Environment-Specific Configurations

#### Development (`dev`)
```yaml
location: "centralindia"
aks_node_pool_vm_size: "Standard_D2s_v4"
deploy_observability_tools: "true"
deploy_azure_container_registry: "true"
deploy_azure_servicebus: "true"
deploy_azure_cosmosdb: "false"
deploy_azure_openai: "false"
```

#### UAT (`uat`)
```yaml
location: "westeurope"
aks_node_pool_vm_size: "Standard_D4s_v4"
deploy_observability_tools: "true"
deploy_azure_container_registry: "true"
deploy_azure_servicebus: "true"
deploy_azure_cosmosdb: "true"
deploy_azure_openai: "true"
```

#### Production (`prod`)
```yaml
location: "eastus" # Multi-region setup
aks_node_pool_vm_size: "Standard_D8s_v4"
deploy_observability_tools: "true"
deploy_azure_container_registry: "true"
deploy_azure_servicebus: "true"
deploy_azure_cosmosdb: "true"
deploy_azure_openai: "true"
```

## GitHub Actions Workflow

### Triggers
- **Push Events**: `main`, `dev`, `uat` branches
- **Pull Requests**: `main` branch (for reviews)
- **Manual Dispatch**: Available for emergency deployments

### Jobs Pipeline
1. **Terraform Plan**: Validates infrastructure changes
2. **Security Scan**: Analyzes security vulnerabilities
3. **Helm Lint**: Validates Kubernetes manifests
4. **Environment Deployment**: Deploys to appropriate environment
5. **Application Deployment**: Deploys applications via Helm

## Getting Started

### For Developers
1. Create feature branch from `dev`
2. Make changes to `infra/` directory
3. Create PR to `dev` branch
4. After merge, changes auto-deploy to dev environment

### For QA/Testing
1. Create branch from `uat`
2. Test infrastructure changes
3. Create PR to `uat` branch
4. After merge, changes auto-deploy to UAT environment

### For Production Releases
1. Create PR from `uat` to `main`
2. Review and approve PR
3. Merge to `main` branch
4. **Manual approval required** for production deployment
5. Monitor deployment in GitHub Actions

## Security Considerations

### Production Safeguards
- Manual approval required for all production deployments
- Separate Azure credentials for production (`AZURE_CREDENTIALS_PROD`)
- Enhanced monitoring and logging
- Multi-region deployment for high availability

### Environment Isolation
- Separate resource groups per environment
- Environment-specific secrets and configurations
- Network isolation between environments

## Monitoring & Observability

### Each Environment Includes
- Azure Monitor and Application Insights
- Log Analytics Workspace
- Container Insights for AKS
- Cost tracking and alerts

### Production Enhancements
- Multi-region monitoring
- Enhanced alerting
- Compliance reporting
- Performance dashboards

## Troubleshooting

### Common Issues
1. **Deployment Failures**: Check GitHub Actions logs
2. **Permission Issues**: Verify Azure credentials
3. **Resource Conflicts**: Check resource naming conventions
4. **Approval Delays**: Production deployments require manual approval

### Support
- Check GitHub Actions workflow logs
- Review Terraform plan outputs
- Monitor Azure portal for resource status
- Contact DevOps team for production issues

## Best Practices

1. **Always test in dev first**
2. **Use descriptive commit messages**
3. **Review Terraform plans before merging**
4. **Monitor costs in each environment**
5. **Keep production changes minimal and well-tested**
6. **Use feature flags for application changes**

---

*This strategy ensures safe, automated deployments while maintaining proper separation between development, testing, and production environments.*
