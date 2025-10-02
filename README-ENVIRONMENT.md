# 🌍 Multi-Environment Configuration Guide

## Overview

This application supports multiple environments with automatic detection and smart configuration. The system automatically adapts URLs, settings, and behavior based on the deployment environment.

## 🚀 Quick Start

### Local Development
```bash
# Setup local environment
python scripts/setup-environment.py local

# Start application
python scripts/run-local.py
# OR
python main.py
```

### Production Testing
```bash
# Test with production URLs locally
python scripts/run-production-test.py
```

### Validation
```bash
# Validate environment configuration
python scripts/validate-environment.py

# Security check
python scripts/security-check.py
```

## 🏗️ Environment Detection

The application automatically detects the environment using:

1. **Explicit**: `ENVIRONMENT` environment variable
2. **Azure**: `WEBSITE_SITE_NAME` environment variable
3. **Kubernetes**: `KUBERNETES_SERVICE_HOST` environment variable
4. **Docker**: `/.dockerenv` file existence
5. **Default**: Falls back to `local`

## 📁 File Structure

```
ai-crenovent/
├── .env                     # Local development (NEVER COMMIT)
├── .env.production         # Production config (NEVER COMMIT)
├── .env.template           # Safe template (COMMIT)
├── utils/environment.py    # Environment detection logic
├── config/settings.py      # Pydantic settings with env support
├── scripts/
│   ├── setup-environment.py      # Environment setup
│   ├── run-local.py              # Local development runner
│   ├── run-production-test.py    # Production test runner
│   ├── validate-environment.py   # Environment validation
│   └── security-check.py         # Security validation
└── docs/
    ├── ENVIRONMENT-SETUP.md      # Detailed setup guide
    └── SECURITY.md               # Security guidelines
```

## 🔧 Environment Variables

### Core Configuration
```env
# Environment Detection
ENVIRONMENT=local|development|staging|production|azure

# Service Configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
AI_SERVICE_URL=http://localhost:8000

# Backend Integration
NODEJS_BACKEND_URL=http://localhost:3001
BACKEND_BASE_URL=http://localhost:3001
```

### Database & Azure
```env
# Database
DATABASE_URL=postgresql://user:pass@host:5432/db

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/

# Azure Services
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
AZURE_TENANT_ID=your_tenant_id
```

## 🌐 Environment Configurations

### Local Development
- **URLs**: `localhost` with standard ports
- **Debug**: Enabled
- **Reload**: Enabled
- **SSL**: Disabled

### Production (Azure)
- **URLs**: Azure App Service URLs
- **Debug**: Disabled
- **Reload**: Disabled
- **SSL**: Enabled

## 🔒 Security Features

### Automatic Protection
- ✅ Comprehensive `.gitignore` for all secret files
- ✅ Environment variable validation
- ✅ Security scanning scripts
- ✅ Template files for safe sharing

### Secret Management
- ✅ No hardcoded secrets in source code
- ✅ Environment-specific configuration
- ✅ Azure Key Vault integration ready
- ✅ Automatic secret detection

## 🛠️ Development Workflows

### 1. Initial Setup
```bash
# Clone repository
git clone <repository-url>
cd ai-crenovent

# Setup local environment
python scripts/setup-environment.py local

# Edit .env with your values
nano .env

# Validate setup
python scripts/validate-environment.py
```

### 2. Local Development
```bash
# Start with local configuration
python scripts/run-local.py

# Or use standard method
python main.py
```

### 3. Production Testing
```bash
# Test with production URLs locally
python scripts/run-production-test.py

# Validate production configuration
ENVIRONMENT=production python scripts/validate-environment.py
```

### 4. Deployment
```bash
# Push to main branch (triggers GitHub Actions)
git push origin main

# Or manual deployment
docker build -t myapp .
docker push acrnewcrenoaiapp.azurecr.io/revai-ai-mainv2:latest
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Environment Variables Not Loading
```bash
# Check if .env exists
ls -la .env

# Validate environment
python scripts/validate-environment.py

# Recreate from template
cp .env.template .env
```

#### 2. Wrong URLs Being Used
```bash
# Check environment detection
python -c "from utils.environment import env_config; print(env_config.get_config_summary())"

# Force environment
export ENVIRONMENT=local
python main.py
```

#### 3. Secrets Exposed in Git
```bash
# Check tracked files
git ls-files .env*

# Remove from git
git rm --cached .env
git commit -m "Remove secrets"

# Run security check
python scripts/security-check.py
```

### Debug Commands
```bash
# Show current environment configuration
python -c "
from utils.environment import env_config
import json
print(json.dumps(env_config.get_config_summary(), indent=2))
"

# Test environment detection
python -c "
from utils.environment import get_environment, get_backend_url
print(f'Environment: {get_environment()}')
print(f'Backend URL: {get_backend_url()}')
"

# Validate all settings
python -c "
from config.settings import settings
print(f'Backend URL: {settings.backend_base_url}')
print(f'Service Host: {settings.service_host}')
print(f'Service Port: {settings.service_port}')
"
```

## 📊 Monitoring & Validation

### Automated Checks
- **Environment Validation**: Checks configuration completeness
- **Security Scanning**: Detects exposed secrets
- **Connectivity Testing**: Validates service URLs
- **Git Safety**: Ensures no secrets are tracked

### Manual Verification
```bash
# Full environment check
python scripts/validate-environment.py

# Security audit
python scripts/security-check.py

# Test connectivity
curl http://localhost:8000/health
curl https://revai-api-mainv2.azurewebsites.net/health
```

## 🎯 Best Practices

### Development
- ✅ Always use environment variables for configuration
- ✅ Test with production URLs before deployment
- ✅ Validate environment setup regularly
- ✅ Keep `.env.template` updated

### Security
- ✅ Never commit `.env` files
- ✅ Use different secrets per environment
- ✅ Rotate secrets regularly
- ✅ Run security checks before commits

### Deployment
- ✅ Use GitHub Actions for automated deployment
- ✅ Set environment variables in Azure App Service
- ✅ Monitor application logs after deployment
- ✅ Validate health endpoints

## 📞 Support

For environment setup issues:
1. Check this guide and `ENVIRONMENT-SETUP.md`
2. Run validation scripts
3. Check `SECURITY.md` for security guidelines
4. Contact the development team

## 🔄 Migration Guide

### From Hardcoded URLs
1. Replace hardcoded URLs with `os.getenv()` calls
2. Add environment variables to `.env.template`
3. Update configuration in Azure App Service
4. Test with both local and production URLs

### Adding New Environments
1. Add environment detection logic in `utils/environment.py`
2. Update configuration mappings
3. Create environment-specific scripts
4. Update documentation