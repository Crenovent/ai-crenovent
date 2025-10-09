# CI/CD Integration Guide

## Overview
This guide explains how the PLAT-CICD-003 templates integrate with your existing `deployment.yml` workflow.

## Current Workflow Architecture

### 1. Existing Production Deployment (`deployment.yml`)
- **Triggers**: `main` branch pushes
- **Target**: Azure App Service
- **Registry**: `acrnewcrenoaiapp.azurecr.io`
- **Container**: `revai-ai-mainv2`
- **Service**: `RevAI-AI-mainV2`

### 2. New Microservice CI/CD (`microservice-ci-cd.yml`)
- **Triggers**: `develop` branch pushes, PRs to `main`/`develop`
- **Target**: Kubernetes (AKS) or Azure App Service fallback
- **Registry**: `acrrevaiprod.azurecr.io`
- **Focus**: Microservices in `/microservices/` directory

### 3. Policy Pack CI/CD (`policy-ci-cd.yml`)
- **Triggers**: Changes to `/policies/`, `/tests/`, `policy-harness.yaml`
- **Target**: Policy validation and testing
- **Focus**: OPA policies and security validation

## Integration Strategy

### No Conflicts
✅ **Separate Triggers**: Different path patterns prevent conflicts
- `deployment.yml`: Triggers on root-level changes
- `microservice-ci-cd.yml`: Triggers on `microservices/**` and `helm/**`
- `policy-ci-cd.yml`: Triggers on `policies/**` and `tests/**`

✅ **Different Environments**: 
- `deployment.yml`: Production (Azure App Service)
- `microservice-ci-cd.yml`: Staging (Kubernetes/App Service fallback)

✅ **Different Registries**:
- `deployment.yml`: `acrnewcrenoaiapp.azurecr.io`
- `microservice-ci-cd.yml`: `acrrevaiprod.azurecr.io`

## Usage Scenarios

### Scenario 1: Main Application Updates
```bash
# Edit files in root directory (src/, main.py, etc.)
git add .
git commit -m "Update main application"
git push origin main
# → Triggers deployment.yml only
```

### Scenario 2: Microservice Development
```bash
# Create microservice
mkdir -p microservices/user-service
# Add microservice code
git add microservices/
git commit -m "Add user microservice"
git push origin develop
# → Triggers microservice-ci-cd.yml only
```

### Scenario 3: Policy Updates
```bash
# Update policies
git add policies/
git commit -m "Update access policies"
git push origin main
# → Triggers policy-ci-cd.yml only
```

## Microservice Structure

### Directory Layout
```
microservices/
├── user-service/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       └── index.js
├── order-service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       └── main.py
└── notification-service/
    ├── Dockerfile
    ├── go.mod
    └── src/
        └── main.go
```

### Supported Languages
- **Node.js**: `package.json` + `src/`
- **Python**: `requirements.txt` + `src/`
- **Java**: `pom.xml` + `src/`
- **Go**: `go.mod` + `src/`

## Deployment Flow

### Development → Staging
1. Developer pushes to `develop` branch
2. `microservice-ci-cd.yml` triggers
3. Builds microservice container
4. Pushes to `acrrevaiprod.azurecr.io`
5. Deploys to Kubernetes (if AKS exists) or App Service fallback

### Staging → Production
1. Merge `develop` → `main`
2. `deployment.yml` triggers (existing workflow)
3. Deploys main application to Azure App Service
4. Microservices can be deployed separately if needed

## Configuration

### Environment Variables
```yaml
# microservice-ci-cd.yml
env:
  REGISTRY: acrrevaiprod.azurecr.io
  IMAGE_NAME: microservice-template
  SERVICE_NAME: microservice-template

# deployment.yml (existing)
env:
  AZURE_CONTAINER_REGISTRY: acrnewcrenoaiapp.azurecr.io
  CONTAINER_NAME: revai-ai-mainv2
```

### Registry Credentials
Both workflows use the same Azure credentials but different registries:
- `deployment.yml`: Uses ACR admin credentials
- `microservice-ci-cd.yml`: Uses Azure login with service principal

## Testing the Integration

### 1. Test Microservice CI/CD
```bash
# Create a test microservice
mkdir -p microservices/test-service
echo '{"name": "test-service", "version": "1.0.0"}' > microservices/test-service/package.json

# Push to develop branch
git checkout develop
git add microservices/
git commit -m "Add test microservice"
git push origin develop
```

### 2. Test Policy CI/CD
```bash
# Create a test policy
echo 'package test' > policies/test.rego
echo 'test_always_true { true }' >> policies/test.rego

# Push to any branch
git add policies/
git commit -m "Add test policy"
git push origin main
```

### 3. Test Existing Deployment
```bash
# Make a change to main application
echo '# Test change' >> README.md
git add README.md
git commit -m "Test main app deployment"
git push origin main
```

## Monitoring and Troubleshooting

### Workflow Status
- Check GitHub Actions tab for workflow runs
- Each workflow runs independently
- Failed workflows don't affect others

### Common Issues
1. **Registry Authentication**: Ensure both registries have proper credentials
2. **Path Conflicts**: Verify path filters are correct
3. **Resource Limits**: Check Azure quotas for both registries

### Debugging
```bash
# Check workflow triggers
gh run list --workflow=deployment.yml
gh run list --workflow=microservice-ci-cd.yml
gh run list --workflow=policy-ci-cd.yml

# View specific run
gh run view <run-id>
```

## Future Enhancements

### 1. Unified Registry
- Migrate to single registry (`acrrevaiprod.azurecr.io`)
- Update `deployment.yml` to use same registry

### 2. Advanced GitOps
- Implement ArgoCD for Kubernetes deployments
- Add progressive delivery features

### 3. Cross-Service Communication
- Add service mesh (Istio) for microservice communication
- Implement distributed tracing

## Summary

✅ **No Conflicts**: Workflows are completely separate
✅ **Complementary**: Each serves different purposes
✅ **Scalable**: Easy to add more microservices
✅ **Maintainable**: Clear separation of concerns

The integration provides a robust CI/CD foundation that supports both your existing application and future microservices without any conflicts or disruptions.
