# ================================================================================
# Branch Protection Configuration
# ================================================================================
# This document describes the branch protection rules for the repository
# ================================================================================

## Branch Protection Rules

### Main Branch (Production)
- **Required Status Checks**: 
  - `ci-build-test` (Microservice CI/CD Pipeline)
  - `policy-unit-tests` (Policy Pack CI/CD Pipeline)
  - `policy-harness` (Policy Pack CI/CD Pipeline)
- **Require Up-to-Date Branches**: Yes
- **Require Linear History**: Yes
- **Restrict Pushes**: Yes (only via pull requests)
- **Required Reviewers**: 2
- **Dismiss Stale Reviews**: Yes
- **Require Review from Code Owners**: Yes

### Develop Branch (Staging)
- **Required Status Checks**:
  - `ci-build-test` (Microservice CI/CD Pipeline)
  - `policy-unit-tests` (Policy Pack CI/CD Pipeline)
- **Require Up-to-Date Branches**: Yes
- **Require Linear History**: No
- **Restrict Pushes**: Yes (only via pull requests)
- **Required Reviewers**: 1
- **Dismiss Stale Reviews**: Yes

### Feature Branches
- **Required Status Checks**:
  - `ci-build-test` (Microservice CI/CD Pipeline)
  - `policy-unit-tests` (Policy Pack CI/CD Pipeline)
- **Require Up-to-Date Branches**: Yes
- **Require Linear History**: No
- **Restrict Pushes**: No
- **Required Reviewers**: 1

## Code Owners

Create a `.github/CODEOWNERS` file with:

```
# Global owners
* @devops-team @backend-team @frontend-team

# Infrastructure
/infra/ @devops-team
/.github/workflows/ @devops-team

# Policies
/policies/ @devops-team @security-team
/tests/ @devops-team @security-team

# Microservices
/src/ @backend-team
/helm/ @devops-team @backend-team

# Frontend
/frontend/ @frontend-team
```

## Required Status Checks

### Microservice CI/CD Pipeline
1. **ci-build-test**: Build, test, and security scan
2. **cd-deploy-staging**: Deploy to staging (develop branch only)
3. **cd-deploy-production**: Deploy to production (main branch only)

### Policy Pack CI/CD Pipeline
1. **policy-lint**: Policy syntax validation and linting
2. **policy-unit-tests**: Policy unit tests
3. **policy-harness**: Policy evaluation harness
4. **policy-security-scan**: Security analysis
5. **policy-docs**: Documentation generation

## Implementation Steps

1. **Enable Branch Protection**:
   ```bash
   # Via GitHub CLI
   gh api repos/:owner/:repo/branches/main/protection \
     --method PUT \
     --field required_status_checks='{"strict":true,"contexts":["ci-build-test","policy-unit-tests","policy-harness"]}' \
     --field enforce_admins=true \
     --field required_pull_request_reviews='{"required_approving_review_count":2,"dismiss_stale_reviews":true}' \
     --field restrictions='{"users":[],"teams":[]}'
   ```

2. **Create CODEOWNERS File**:
   ```bash
   mkdir -p .github
   cat > .github/CODEOWNERS << 'EOF'
   # Global owners
   * @devops-team @backend-team @frontend-team
   
   # Infrastructure
   /infra/ @devops-team
   /.github/workflows/ @devops-team
   
   # Policies
   /policies/ @devops-team @security-team
   /tests/ @devops-team @security-team
   
   # Microservices
   /src/ @backend-team
   /helm/ @devops-team @backend-team
   
   # Frontend
   /frontend/ @frontend-team
   EOF
   ```

3. **Configure Required Status Checks**:
   - Go to Repository Settings → Branches
   - Select branch to protect
   - Enable "Require status checks to pass before merging"
   - Select required status checks
   - Enable "Require branches to be up to date before merging"

## Verification

1. **Test Branch Protection**:
   - Create a feature branch
   - Make changes that fail CI
   - Try to merge via PR → Should be blocked
   - Fix issues and retry → Should succeed

2. **Test Policy Validation**:
   - Create a policy with syntax errors
   - Try to merge → Should be blocked by policy-lint
   - Create a policy with failing tests
   - Try to merge → Should be blocked by policy-unit-tests

3. **Test Code Review Requirements**:
   - Create PR without required reviewers
   - Try to merge → Should be blocked
   - Add required reviewers and get approval
   - Try to merge → Should succeed

## Rollback Procedures

1. **Disable Branch Protection**:
   ```bash
   gh api repos/:owner/:repo/branches/main/protection \
     --method DELETE
   ```

2. **Revert CODEOWNERS**:
   ```bash
   git revert <commit-hash>
   ```

3. **Emergency Override**:
   - Repository admins can override branch protection
   - Use sparingly and document reasons
