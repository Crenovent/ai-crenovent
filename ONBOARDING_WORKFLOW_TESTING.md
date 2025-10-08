# Complete End-to-End Onboarding Workflow Testing Guide

## 🚀 Overview
This guide covers testing the complete RevOps user onboarding workflow where users can upload CSV files, have them processed by AI agents, and save the results to the database.

## 📋 Prerequisites
1. **Database Setup**: PostgreSQL with the correct schema
2. **Services Running**: AI Backend, Node.js Backend, Frontend
3. **Authentication**: Valid RevOps Manager user account

## 🔧 Service Startup Order

### 1. Start AI Backend (Python/FastAPI)
```bash
cd ai-crenovent
python main.py
```
- Should start on `http://localhost:8000`
- Verify with: `curl http://localhost:8000/api/onboarding/test-endpoint`

### 2. Start Node.js Backend
```bash
cd backend-crenovent
npm run dev
```
- Should start on `http://localhost:3001`
- Verify with: `curl http://localhost:3001/api/health`

### 3. Start Frontend
```bash
cd frontend-crenovent
npm run dev
```
- Should start on `http://localhost:3000`
- Navigate to RevOps Engineering page

## 📊 Test CSV File
Use the provided sample CSV file: `ai-crenovent/test_data/sample_users.csv`

### Sample CSV Format:
```csv
Name,Email,Role Title,Department,Manager,Location,Reports To Email,Team,Employee Number,Hire Date,User Status,Permissions,User Type
John Smith,john.smith@company.com,Sales Manager,Sales,,New York,,Sales Team,EMP001,2023-01-15,Active,Full Access,Manager
Jane Doe,jane.doe@company.com,VP Sales,Sales,,San Francisco,,Executive Team,EMP002,2022-06-01,Active,Full Access,Executive
Mike Johnson,mike.johnson@company.com,Sales Rep,Sales,john.smith@company.com,Chicago,john.smith@company.com,Sales Team,EMP003,2023-03-10,Active,Standard Access,Individual Contributor
```

## 🧪 End-to-End Testing Steps

### Step 1: Access RevOps Engineering Page
1. Login as RevOps Manager
2. Navigate to RevOps Engineering page
3. Click "Onboarding Agent" button

### Step 2: Upload CSV File
1. Select "Upload CSV" tab
2. Click "Choose File" and select `sample_users.csv`
3. Verify file is loaded (should show file name)
4. Click "Upload & Process"

### Step 3: Review Processed Data
1. Wait for processing to complete
2. Review the field mappings and assignments:
   - **Region**: Assigned based on location
   - **Segment**: Assigned based on department
   - **Level**: Assigned based on role title
   - **Territory**: Assigned based on team
   - **Modules**: Assigned based on role and permissions

### Step 4: Complete Setup
1. Click "Complete Setup" button
2. Wait for database save to complete
3. Verify success message
4. Check database for created users

## 🔍 Verification Points

### Frontend Verification
- ✅ CSV file uploads successfully
- ✅ Processing shows progress indicators
- ✅ Field mappings display correctly
- ✅ Complete setup shows success message
- ✅ Modal closes after completion

### Backend Verification
- ✅ `/api/onboarding/execute-workflow` processes CSV
- ✅ AI agent assigns fields correctly
- ✅ `/api/onboarding/complete-setup` saves to database
- ✅ Users created with proper relationships

### Database Verification
```sql
-- Check created users
SELECT id, email, first_name, last_name, status, reports_to 
FROM app_auth.users 
WHERE tenant_id = 'your-tenant-id';

-- Check role assignments
SELECT u.email, r.name as role_name
FROM app_auth.users u
JOIN app_auth.users_role ur ON u.id = ur.user_id
JOIN app_auth.roles r ON ur.role_id = r.id
WHERE u.tenant_id = 'your-tenant-id';

-- Check password reset tokens
SELECT tenant_email, status, expires_at
FROM app_auth.password_reset_tokens
WHERE tenant_id = 'your-tenant-id';
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CSV Upload Fails
- **Check**: File format (must be CSV)
- **Check**: File size (should be < 10MB)
- **Check**: Required columns (Name, Email)

#### 2. Processing Fails
- **Check**: AI backend is running
- **Check**: Database connection
- **Check**: Logs in AI backend console

#### 3. Complete Setup Fails
- **Check**: Database schema matches
- **Check**: Tenant ID is valid
- **Check**: User permissions

#### 4. Database Errors
- **Check**: PostgreSQL is running
- **Check**: Schema is created
- **Check**: RLS policies are enabled

### Debug Commands
```bash
# Check AI backend logs
tail -f ai-crenovent/logs/onboarding.log

# Check database connection
psql -h localhost -U your_user -d your_db -c "SELECT 1;"

# Test API endpoints
curl -X POST http://localhost:8000/api/onboarding/test-endpoint
curl -X GET http://localhost:3001/api/health
```

## 📈 Expected Results

### Successful Workflow
1. **7 users** processed from CSV
2. **Field assignments** completed:
   - Regions: New York, San Francisco, Chicago, Boston, Austin
   - Segments: Sales, Marketing, Human Resources
   - Levels: Manager, Executive, Individual Contributor
   - Territories: Sales Team, Executive Team, Marketing Team, HR Team
   - Modules: Assigned based on role and permissions

3. **Database records** created:
   - Users table: 7 new records
   - Roles table: Default 'user' role created
   - User-role assignments: 7 assignments
   - Password reset tokens: 7 tokens for invite emails

4. **Manager relationships** established:
   - Mike Johnson → John Smith
   - Tom Brown → Sarah Wilson
   - Bob Miller → Lisa Davis

## 🎯 Success Criteria
- ✅ All users from CSV are created in database
- ✅ Field assignments are accurate and logical
- ✅ Manager relationships are properly established
- ✅ Roles are assigned correctly
- ✅ Password reset tokens are generated
- ✅ Frontend shows success message
- ✅ No errors in any service logs

## 🔄 Next Steps After Testing
1. **Email Integration**: Configure email service for invite emails
2. **Role Customization**: Create custom roles based on job titles
3. **Module Assignment**: Refine module assignment logic
4. **Validation Rules**: Add more validation for CSV data
5. **Performance**: Optimize for larger CSV files

---

**Note**: This workflow demonstrates the complete automation continuum from RBA (Rule-Based Automation) to intelligent field assignment, showcasing the platform's ability to handle complex user onboarding scenarios with minimal manual intervention.
