# 🔒 Security Guidelines

## Environment Files & Secrets Management

### ⚠️ CRITICAL: Never Commit These Files
The following files contain sensitive information and should **NEVER** be committed to git:

```
.env                          # Local development secrets
.env.production              # Production secrets  
.env.development             # Development secrets
.env.staging                 # Staging secrets
.env.azure                   # Azure-specific secrets
azure-app-settings.json      # Azure App Service settings
*.secrets                    # Any secrets files
secrets/                     # Secrets directory
```

### ✅ Safe to Commit
These template files are safe to commit as they contain no actual secrets:

```
.env.template                # Environment template
.env.example                 # Example configuration
azure-app-settings.template.json  # Azure settings template
```

## Environment Setup

### 1. Local Development
```bash
# Copy template and fill in your values
cp .env.template .env
# Edit .env with your local configuration
```

### 2. Production Deployment
```bash
# Copy template and fill in production values
cp .env.template .env.production
# Edit .env.production with production configuration
```

### 3. Azure App Service
Set environment variables directly in Azure Portal:
- Go to App Service → Configuration → Application Settings
- Add each environment variable individually
- Never upload .env files to Azure

## Secret Rotation

### Regular Tasks
- [ ] Rotate Azure OpenAI API keys every 90 days
- [ ] Rotate database passwords every 90 days  
- [ ] Rotate JWT secrets every 30 days
- [ ] Review and rotate service principal secrets every 180 days

### Emergency Rotation
If secrets are compromised:
1. Immediately rotate all affected secrets
2. Update all environments (local, staging, production)
3. Restart all services
4. Review access logs

## Best Practices

### ✅ DO
- Use environment variables for all secrets
- Use Azure Key Vault for production secrets
- Use different secrets for each environment
- Regularly rotate secrets
- Use strong, unique passwords
- Enable MFA on all Azure accounts

### ❌ DON'T
- Hardcode secrets in source code
- Commit .env files to git
- Share secrets via email/chat
- Use the same secrets across environments
- Store secrets in plain text files
- Use weak or default passwords

## Incident Response

If you accidentally commit secrets:
1. **Immediately** rotate the compromised secrets
2. Remove the secrets from git history:
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch .env' \
   --prune-empty --tag-name-filter cat -- --all
   ```
3. Force push to remote repository
4. Notify the team
5. Update all environments with new secrets

## Environment Variables Reference

### Required for All Environments
```
DATABASE_URL=postgresql://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
```

### Environment-Specific
```
# Local
BACKEND_BASE_URL=http://localhost:3001
ENVIRONMENT=local

# Production  
BACKEND_BASE_URL=https://revai-api-mainv2.azurewebsites.net
ENVIRONMENT=production
```

## Monitoring & Alerts

Set up alerts for:
- Failed authentication attempts
- Unusual API usage patterns
- Secret rotation reminders
- Security configuration changes

## Contact

For security issues or questions:
- Create a private issue in the repository
- Contact the development team directly
- For critical security issues, follow the incident response process