# Contributing to RevAI Infrastructure

Thank you for your interest in contributing to the RevAI infrastructure repository! This guide will help you understand our development process and how to contribute effectively.

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following tools installed:

- [Terraform](https://www.terraform.io/downloads.html) >= 1.9.2
- [Helm](https://helm.sh/docs/intro/install/) >= 3.12.0
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) >= 2.50.0
- [kubectl](https://kubernetes.io/docs/tasks/tools/) >= 1.28.0
- [Git](https://git-scm.com/downloads) >= 2.30.0

### Development Environment Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/your-username/infra.git
   cd infra
   ```

2. **Set up Azure Authentication**:
   ```bash
   az login
   az account set --subscription "your-subscription-id"
   ```

3. **Configure Environment Variables**:
   ```bash
   export ARM_CLIENT_ID="your-client-id"
   export ARM_CLIENT_SECRET="your-client-secret"
   export ARM_SUBSCRIPTION_ID="your-subscription-id"
   export ARM_TENANT_ID="your-tenant-id"
   ```

4. **Test Your Setup**:
   ```bash
   cd terraform/environments/dev
   terraform init
   terraform plan -var-file="dev.tfvars"
   ```

## 📋 Development Workflow

### 1. Branch Strategy

We follow the GitFlow branching model:

- **`main`**: Production-ready code
- **`develop`**: Integration branch for features
- **`feature/*`**: Feature development branches
- **`hotfix/*`**: Critical production fixes
- **`release/*`**: Release preparation branches

### 2. Making Changes

1. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**:
   - Update Terraform modules
   - Modify Helm charts
   - Update documentation
   - Add tests

3. **Test Your Changes**:
   ```bash
   # Terraform validation
   terraform validate
   terraform plan
   
   # Helm validation
   helm lint ./helm/orchestrator
   helm template orchestrator ./helm/orchestrator
   ```

4. **Commit Your Changes**:
   ```bash
   git add .
   git commit -m "feat: add new infrastructure component"
   ```

### 3. Pull Request Process

1. **Push Your Branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request**:
   - Use the PR template
   - Provide clear description
   - Link related issues
   - Request appropriate reviewers

3. **Address Review Feedback**:
   - Make requested changes
   - Respond to comments
   - Update tests if needed

4. **Merge After Approval**:
   - Ensure CI passes
   - Get required approvals
   - Merge via GitHub UI

## 🧪 Testing Guidelines

### Terraform Testing

1. **Format Check**:
   ```bash
   terraform fmt -check -recursive
   ```

2. **Validation**:
   ```bash
   terraform validate
   ```

3. **Plan Review**:
   ```bash
   terraform plan -var-file="dev.tfvars"
   ```

4. **Security Scanning**:
   ```bash
   tfsec ./terraform
   checkov -d ./terraform
   ```

### Helm Testing

1. **Lint Charts**:
   ```bash
   helm lint ./helm/orchestrator
   helm lint ./helm/evidence-worker
   ```

2. **Template Testing**:
   ```bash
   helm template orchestrator ./helm/orchestrator
   helm template evidence-worker ./helm/evidence-worker
   ```

3. **Dry Run Deployment**:
   ```bash
   helm install orchestrator ./helm/orchestrator --dry-run --debug
   ```

## 📝 Code Standards

### Terraform Standards

1. **Naming Conventions**:
   - Use snake_case for resource names
   - Prefix resources with environment: `dev-`, `staging-`, `prod-`
   - Use descriptive names: `aks-cluster`, `postgresql-database`

2. **File Organization**:
   - One resource per file for complex resources
   - Group related resources in modules
   - Use consistent variable naming

3. **Documentation**:
   - Add descriptions to all variables
   - Include usage examples
   - Document module dependencies

### Helm Standards

1. **Chart Structure**:
   - Follow Helm best practices
   - Use consistent naming conventions
   - Include proper labels and selectors

2. **Values Management**:
   - Provide sensible defaults
   - Use environment-specific values files
   - Document all configurable options

3. **Security**:
   - Use non-root containers
   - Enable security contexts
   - Implement proper RBAC

## 🔒 Security Guidelines

### Secret Management

1. **Never Commit Secrets**:
   - Use Azure Key Vault for secrets
   - Use GitHub Secrets for CI/CD
   - Use environment variables for local development

2. **Access Control**:
   - Follow principle of least privilege
   - Use managed identities where possible
   - Implement proper RBAC policies

3. **Network Security**:
   - Use private endpoints
   - Implement network security groups
   - Enable firewall rules

### Compliance

1. **Data Protection**:
   - Encrypt data at rest and in transit
   - Implement proper backup strategies
   - Follow data retention policies

2. **Audit Logging**:
   - Enable comprehensive logging
   - Implement log retention
   - Monitor for security events

## 🚨 Emergency Procedures

### Incident Response

1. **Critical Issues**:
   - Create hotfix branch from main
   - Implement minimal fix
   - Test thoroughly
   - Deploy immediately

2. **Rollback Procedures**:
   - Document rollback steps
   - Test rollback procedures
   - Maintain rollback capability

### Communication

1. **Incident Notification**:
   - Use #devops Slack channel
   - Email devops@crenovent.com
   - Update status page if needed

2. **Post-Incident**:
   - Conduct post-mortem
   - Document lessons learned
   - Implement improvements

## 📚 Documentation Standards

### Required Documentation

1. **Module Documentation**:
   - README.md for each module
   - Usage examples
   - Input/output variables
   - Dependencies

2. **Environment Documentation**:
   - Environment-specific README
   - Deployment procedures
   - Configuration options
   - Troubleshooting guides

3. **API Documentation**:
   - OpenAPI specifications
   - Endpoint documentation
   - Authentication requirements
   - Rate limiting information

### Documentation Format

1. **Markdown Standards**:
   - Use proper heading hierarchy
   - Include table of contents
   - Use code blocks with syntax highlighting
   - Include diagrams where helpful

2. **Code Comments**:
   - Explain complex logic
   - Document assumptions
   - Include TODO items
   - Reference external documentation

## 🤝 Community Guidelines

### Code of Conduct

1. **Be Respectful**:
   - Use inclusive language
   - Be constructive in feedback
   - Respect different perspectives

2. **Be Professional**:
   - Focus on technical merit
   - Provide constructive criticism
   - Help others learn and grow

3. **Be Collaborative**:
   - Share knowledge and expertise
   - Help with code reviews
   - Mentor new contributors

### Getting Help

1. **Documentation**:
   - Check existing documentation first
   - Search GitHub issues
   - Review pull request discussions

2. **Community Support**:
   - Use GitHub Discussions
   - Join #devops Slack channel
   - Attend team meetings

3. **Direct Support**:
   - Email devops@crenovent.com
   - Create GitHub issue
   - Request code review

## 📈 Performance Guidelines

### Resource Optimization

1. **Cost Management**:
   - Use appropriate instance sizes
   - Implement auto-scaling
   - Monitor resource usage
   - Optimize storage costs

2. **Performance**:
   - Use CDN for static content
   - Implement caching strategies
   - Optimize database queries
   - Monitor performance metrics

### Monitoring

1. **Observability**:
   - Implement comprehensive logging
   - Use structured logging
   - Include correlation IDs
   - Monitor key metrics

2. **Alerting**:
   - Set up meaningful alerts
   - Use appropriate thresholds
   - Implement escalation procedures
   - Test alerting systems

## 🎯 Release Process

### Version Management

1. **Semantic Versioning**:
   - Use semantic versioning (MAJOR.MINOR.PATCH)
   - Update version numbers appropriately
   - Tag releases properly

2. **Release Notes**:
   - Document breaking changes
   - List new features
   - Include bug fixes
   - Note security updates

### Deployment Process

1. **Environment Promotion**:
   - Dev → Staging → Production
   - Test each environment thoroughly
   - Use feature flags where appropriate
   - Implement canary deployments

2. **Rollback Strategy**:
   - Maintain rollback capability
   - Test rollback procedures
   - Document rollback steps
   - Monitor for issues

---

## 📞 Contact Information

- **DevOps Team**: devops@crenovent.com
- **Slack**: #devops channel
- **GitHub**: Create issues for bugs and feature requests
- **Emergency**: Use #devops-alerts Slack channel

Thank you for contributing to RevAI infrastructure! Your contributions help us build a more robust, secure, and scalable platform.
