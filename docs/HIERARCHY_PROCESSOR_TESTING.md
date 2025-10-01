# 🧪 Hierarchy Processor - Complete Testing Guide

## 🚀 **Quick Start Testing**

### **Prerequisites**
```bash
# 1. Install Python dependencies
cd crenovent-ai-service
pip install -r requirements_hierarchy.txt

# 2. Ensure Node.js dependencies are installed  
cd ../crenovent-backend
npm install

# 3. Ensure your PostgreSQL database is running
```

### **Start Both Services**

**Terminal 1 - Python AI Service (Port 8000):**
```bash
cd crenovent-ai-service
python main.py
```

**Terminal 2 - Node.js Backend (Port 3000):**
```bash
cd crenovent-backend
npm start
```

## 🔧 **API Testing**

### **1. Test Python Service Directly**

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Get Available HRMS Systems:**
```bash
curl http://localhost:8000/api/hierarchy/systems
```

**Detect System from Headers:**
```bash
curl "http://localhost:8000/api/hierarchy/detect-system?headers=Full%20Name,Email,Manager%20Email,Department"
```

**Test CSV Normalization (Direct):**
```bash
curl -X POST http://localhost:8000/api/hierarchy/normalize-csv \
  -H "Content-Type: application/json" \
  -d '{
    "csv_data": [
      {
        "Full Name": "John Smith",
        "Email": "john.smith@company.com", 
        "Manager Email": "jane.doe@company.com",
        "Department": "Sales",
        "Job Title": "Account Executive"
      }
    ],
    "tenant_id": 1300,
    "uploaded_by": 1323
  }'
```

**Upload CSV File:**
```bash
curl -X POST http://localhost:8000/api/hierarchy/upload-csv \
  -F "file=@test_data/salesforce_sample.csv"
```

### **2. Test Node.js Integration**

The Node.js backend now automatically calls the Python service when processing CSV uploads.

**Test with Frontend Upload or Postman:**
- URL: `http://localhost:3000/api/register/csv` (or your CSV upload endpoint)
- Method: POST
- Content-Type: multipart/form-data
- Body: Upload any of the test CSV files from `test_data/`

## 📊 **Test Data Files**

We've created test CSV files for different HRMS systems:

### **Salesforce Format** (`test_data/salesforce_sample.csv`)
```csv
Full Name,Email,Manager Email,Department,Job Title,Location,Employee ID,Start Date
John Smith,john.smith@company.com,jane.doe@company.com,Sales,Account Executive,New York,EMP001,2023-01-15
```

### **HubSpot Format** (`test_data/hubspot_sample.csv`)  
```csv
Employee Name,Work Email,Reports To Email,Team,Role,Office Location,Employee Number,Hire Date
John Smith,john.smith@company.com,jane.doe@company.com,Revenue,Account Executive,New York Office,001,01/15/2023
```

### **Zoho Format** (`test_data/zoho_sample.csv`)
```csv
Employee Name,Email ID,Reporting Manager Email,Department Name,Designation,Work Location,Employee Code,Date of Joining
John Smith,john.smith@company.com,jane.doe@company.com,Sales Department,Account Executive,New York,ZH001,15-Jan-2023
```

## 🌐 **Frontend Testing**

### **Where to Test on Frontend**

1. **Find CSV Upload Interface:**
   - Look for user management or admin sections
   - Search for "Upload CSV", "Import Users", or "Bulk Upload"
   - Common paths: `/admin/users`, `/settings/users`, `/import`

2. **Test Process:**
   - Login as admin user
   - Navigate to CSV upload section
   - Upload one of the test CSV files
   - Check browser console for processing logs
   - Verify users are created with correct hierarchy

### **Expected Behavior**

**Before Integration:**
- CSV uploads work but field names must match exactly
- Different HRMS formats fail or create incorrect data

**After Integration:**
- Any supported HRMS CSV format works automatically
- System detects format and maps fields correctly
- Console shows normalization success messages
- Users created with proper hierarchy relationships

## 🔍 **Troubleshooting**

### **Common Issues**

**1. Python Service Not Starting:**
```bash
# Check if port 8000 is in use
netstat -an | findstr :8000

# Kill process if needed
taskkill /PID <process_id> /F
```

**2. Node.js Can't Reach Python Service:**
- Verify Python service is running on port 8000
- Check firewall settings
- Ensure both services are on same network

**3. CSV Processing Fails:**
- Check Python service logs for errors
- Verify CSV format matches expected structure
- Check if required fields (Name, Email) are present

**4. Field Mapping Issues:**
- Test detection endpoint to verify system recognition
- Check mapping configuration files in `hierarchy_processor/config/mappings/`
- Add custom mappings if needed

### **Debug Logs**

**Python Service Logs:**
- Shows HRMS detection results
- Field mapping details
- Processing statistics
- Validation errors

**Node.js Backend Logs:**
- CSV normalization success/failure
- Processing summaries
- Fallback behavior

## ✅ **Validation Checklist**

### **Python Service Tests**
- [ ] Health check responds
- [ ] Available systems endpoint works
- [ ] System detection works for all test files
- [ ] CSV normalization returns expected format
- [ ] File upload endpoint processes CSV correctly

### **Node.js Integration Tests**
- [ ] CSV upload calls Python service
- [ ] Graceful fallback when Python service is down
- [ ] Normalized data creates users correctly
- [ ] Hierarchy relationships are preserved
- [ ] RBAC assignments work as expected

### **End-to-End Tests**
- [ ] Upload Salesforce CSV → Users created correctly
- [ ] Upload HubSpot CSV → Users created correctly  
- [ ] Upload Zoho CSV → Users created correctly
- [ ] Upload unknown format → Falls back gracefully
- [ ] Frontend shows success/error messages appropriately

## 📈 **Performance Testing**

### **Load Testing**
```bash
# Test with larger CSV files
# Create test file with 100+ users
# Monitor processing time and memory usage
```

### **Concurrent Testing**
```bash
# Test multiple simultaneous uploads
# Verify no race conditions
# Check database integrity
```

## 🔧 **Configuration**

### **Adding New HRMS Systems**

1. Create new mapping file: `hierarchy_processor/config/mappings/new_system.yaml`
2. Define field mappings following existing patterns
3. Add system detection rules in `detector.py`
4. Test with sample CSV from new system

### **Customizing Field Mappings**

Edit existing YAML files in `hierarchy_processor/config/mappings/` to adjust field mappings for your specific needs.

## 📚 **API Documentation**

### **Python Service Endpoints**

- `GET /health` - Health check
- `GET /api/hierarchy/systems` - List available HRMS systems
- `GET /api/hierarchy/detect-system?headers=...` - Detect HRMS from headers
- `POST /api/hierarchy/normalize-csv` - Normalize CSV data
- `POST /api/hierarchy/upload-csv` - Upload and process CSV file

### **Request/Response Formats**

All endpoints return JSON with comprehensive error handling and validation details.

---

## 🎯 **Success Criteria**

✅ **Integration Complete When:**
- Python service runs without errors
- Node.js backend successfully calls Python service  
- All test CSV formats process correctly
- Frontend CSV upload works with any supported format
- Graceful fallback when Python service is unavailable
- Users and hierarchy created correctly in database
- RBAC permissions assigned properly

**Ready for production use!** 🚀
