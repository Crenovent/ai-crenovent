# RevAI Deployment Setup

## GitHub Actions Deployment

This repository uses GitHub Actions for automated deployment to Azure App Service.

### Prerequisites

1. **Azure Resources** (Already configured):
   - Resource Group: `rg-newCrenoApp`
   - Container Registry: `acrNewcrenoAIapp`
   - App Service: `RevAI-AI-mainV2`

2. **GitHub Secrets Setup**:
   You need to configure the following secret in your GitHub repository:

   **Secret Name**: `AZURE_CREDENTIALS`
   
   **Secret Value** (JSON format):
   ```json
   {
     "clientId": "your-client-id-here",
     "clientSecret": "your-client-secret-here",
     "subscriptionId": "your-subscription-id-here",
     "tenantId": "your-tenant-id-here"
   }
   ```

### How to Set Up GitHub Secrets

1. Go to your GitHub repository: `https://github.com/Crenovent/ai-crenovent`
2. Click on **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Name: `AZURE_CREDENTIALS`
6. Value: Paste the JSON above
7. Click **Add secret**

### Deployment Triggers

The GitHub Actions workflow will automatically trigger on:
- **Push to `main` branch**: Full deployment with tests
- **Push to `develop` branch**: Full deployment
- **Pull requests to `main`**: Build and test only
- **Manual trigger**: Via GitHub Actions UI

### Manual Deployment

If you prefer manual deployment, you can use the deployment scripts:

#### Windows PowerShell:
```powershell
.\deploy.ps1
```

#### Linux/Mac:
```bash
./deploy.sh
```

### Local Development

For local development, copy the secrets template:
```bash
cp azure-secrets.template azure-secrets.env
```

Then edit `azure-secrets.env` with your actual values.

### Security Notes

- Never commit actual secrets to the repository
- The `azure-secrets.template` file is safe to commit (contains placeholders)
- Actual secrets are stored in GitHub repository secrets
- Local secrets files are ignored by Git (see `.gitignore`)

### Troubleshooting

If deployment fails:
1. Check GitHub Actions logs
2. Verify GitHub secrets are configured correctly
3. Ensure Azure resources are accessible
4. Check App Service logs: `az webapp log tail --name RevAI-AI-mainV2 --resource-group rg-newCrenoApp`

### Application URLs

- **Production**: https://revai-ai-mainv2-evc4fzcva8bdcnfm.centralindia-01.azurewebsites.net
- **Health Check**: https://revai-ai-mainv2-evc4fzcva8bdcnfm.centralindia-01.azurewebsites.net/health
- **API Docs**: https://revai-ai-mainv2-evc4fzcva8bdcnfm.centralindia-01.azurewebsites.net/docs
