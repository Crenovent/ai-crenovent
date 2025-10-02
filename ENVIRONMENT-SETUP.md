# 🌍 Environment Setup Guide

## Quick Start

### 1. Local Development Setup
```bash
# Copy environment template
cp .env.template .env

# Edit with your local values
nano .env

# Run setup script (optional)
python scripts/setup-environment.py local
```

### 2. Production Deployment Setup
```bash
# Create production environment file
cp .env.template .env.production

# Edit with production values
nano .env.production

# Set up Azure App Service environment variables
# (See Azure Configuration section below)
```

## Environment Files

### 📁 File Structure
```
ai-crenovent/
├── .env                    # Local development (DO NOT COMMIT)
├── .env.production        # Production config (DO NOT COMMIT)
├── .env.template          # Safe template (COMMIT THIS)
├── .gitignore            # Protects secrets (COMMIT THIS)
└── SECURITY.md           # Security guidelines (COMMIT THIS)
```

### 🔒 Security Rules
- ✅ **COMMIT**: `.env.template`, `.env.example`
- ❌ **NEVER COMMIT**: `.env`, `.env.production`, `.env.*`

## Environment Variables

### 🏠 Local Development
```env
# Service Configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
AI_SERVICE_URL=http://localhost:8000

# Backend Integration
NODEJS_BACKEND_URL=http://localhost:3001
BACKEND_BASE_URL=http://localhost:3001

# Environment
ENVIRONMENT=local
```

### ☁️ Production (Azure)
```env
# Service Configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
AI_SERVICE_URL=https://revai-ai-mainv2-evc4fzcva8bdcnfm.centralindia-01.azurewebsites.net

# Backend Integration
NODEJS_BACKEND_URL=https://revai-api-mainv2.azurewebsites.net
BACKEND_BASE_URL=https://revai-api-mainv2.azurewebsites.net

# Environment
ENVIRONMENT=production
```

## Azure Configuration

### Setting Environment Variables in Azure App Service

#### Method 1: Azure Portal
1. Go to Azure Portal → App Services → Your App
2. Navigate to **Configuration** → **Application settings**
3. Click **+ New application setting**
4. Add each environment variable:
   ```
   Name: NODEJS_BACKEND_URL
   Value: https://revai-api-mainv2.azurewebsites.net
   ```

#### Method 2: Azure CLI
```bash
# Set multiple environment variables
az webapp config appsettings set \
  --resource-group rg-newCrenoApp \
  --name RevAI-AI-mainV2 \
  --settings \
    NODEJS_BACKEND_URL="https://revai-api-mainv2.azurewebsites.net" \
    BACKEND_BASE_URL="https://revai-api-mainv2.azurewebsites.net" \
    SERVICE_HOST="0.0.0.0" \
    WEBSITES_PORT="8000" \
    ENVIRONMENT="production"
```

#### Method 3: GitHub Actions (Automated)
```yaml
# In your deployment workflow
- name: Set App Service Configuration
  run: |
    az webapp config appsettings set \
      --resource-group ${{ env.RESOURCE_GROUP }} \
      --name ${{ env.APP_SERVICE_NAME }} \
      --settings @azure-app-settings.json
```

## Environment Detection

The application automatically detects the environment:

### 🔍 Detection Logic
1. **Azure**: Detects `WEBSITE_SITE_NAME` environment variable
2. **Kubernetes**: Detects `KUBERNETES_SERVICE_HOST`
3. **Docker**: Detects `/.dockerenv` file
4. **Local**: Default fallback

### 🎯 Smart Configuration
```python
from utils.environment import env_config

# Automatically gets the right URL based on environment
backend_url = env_config.get_backend_url()

# Local: http://localhost:3001
# Azure: https://revai-api-mainv2.azurewebsites.net
```

## Development Workflows

### 🏠 Local Development
```bash
# Start local services
python main.py

# The app will automatically use:
# - localhost URLs
# - Debug mode enabled
# - Auto-reload enabled
```

### 🧪 Test Production URLs Locally
```bash
# Set production environment variables temporarily
export NODEJS_BACKEND_URL=https://revai-api-mainv2.azurewebsites.net
export BACKEND_BASE_URL=https://revai-api-mainv2.azurewebsites.net
export ENVIRONMENT=production

# Run the app
python main.py

# The app will use production URLs but run locally
```

### 🚀 Deploy to Azure
```bash
# Push to main branch - GitHub Actions handles deployment
git push origin main

# Or deploy manually
docker build -t myapp .
docker push acrnewcrenoaiapp.azurecr.io/revai-ai-mainv2:latest
```

## Troubleshooting

### ❌ Common Issues

#### 1. "Backend connection failed"
```bash
# Check if backend URL is correct
echo $BACKEND_BASE_URL

# For local development, ensure backend is running
curl http://localhost:3001/health
```

#### 2. "Environment variables not loaded"
```bash
# Check if .env file exists
ls -la .env

# Validate environment setup
python scripts/setup-environment.py validate
```

#### 3. "Secrets exposed in git"
```bash
# Check what's tracked
git ls-files .env*

# Remove from git if needed
git rm --cached .env
git commit -m "Remove secrets from git"
```

### ✅ Validation Commands
```bash
# Check environment setup
python scripts/setup-environment.py validate

# Check .gitignore configuration
python scripts/setup-environment.py check

# Test environment detection
python -c "from utils.environment import env_config; print(env_config.get_config_summary())"
```

## Best Practices

### 🔒 Security
- Use different secrets for each environment
- Rotate secrets regularly
- Never commit `.env` files
- Use Azure Key Vault for production secrets

### 🏗️ Development
- Keep `.env.template` updated
- Document all required environment variables
- Use environment detection for smart defaults
- Test with production URLs before deployment

### 🚀 Deployment
- Set environment variables in Azure App Service
- Use GitHub Actions for automated deployment
- Monitor application logs for configuration issues
- Validate environment after deployment

## Support

For environment setup issues:
1. Check this guide first
2. Run validation scripts
3. Check SECURITY.md for security guidelines
4. Contact the development team