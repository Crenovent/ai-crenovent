# 🧪 Frontend Testing Guide - Hierarchy Processor

## 🎯 **Testing Steps**

### **Step 1: Start Both Services**
```bash
# Terminal 1 - Python AI Service
cd crenovent-ai-service
python main.py  # Port 8000

# Terminal 2 - Node.js Backend  
cd crenovent-backend
npm start  # Port 3000
```

### **Step 2: Access Frontend**
1. **Open browser**: `http://localhost:3000` (or your domain)
2. **Login** as admin user
3. **Navigate**: `/dashboard/user/configure-user`

### **Step 3: Upload Test Files**

**Location**: Look for blue **"Upload CSV"** button in User Management section

**Test Files Available:**
- `crenovent-ai-service/test_data/salesforce_sample.csv`
- `crenovent-ai-service/test_data/hubspot_sample.csv` 
- `crenovent-ai-service/test_data/zoho_sample.csv`
- `crenovent-ai-service/test_data/workday_sample.csv`

## 🔍 **What to Look For**

### **In Browser Console (F12 → Console):**
```
✅ CSV normalized successfully: {
  detectedSystem: "hubspot",
  confidence: 0.762,
  originalRecords: 6,
  normalizedRecords: 6
}

📊 Processing Summary: {
  mappedFields: ["Name", "Email", "Role Title", ...],
  unmappedHeaders: [],
  processingWarnings: []
}
```

### **Success Indicators:**
- ✅ **Upload completes successfully**
- ✅ **Users appear in user list** 
- ✅ **Hierarchy shows correctly** (if you have hierarchy view)
- ✅ **Email notifications sent** (check logs)
- ✅ **Console shows normalization logs**

### **Error Indicators:**
- ❌ **"Hierarchy processor service unavailable"** → Python service not running
- ❌ **Upload fails** → Check both services are running
- ❌ **No users created** → Check Node.js logs

## 🧪 **Testing Scenarios**

### **Test 1: Salesforce Format**
1. Upload `salesforce_sample.csv`
2. **Expected**: Detects as "salesforce", creates 6 users
3. **Hierarchy**: John Smith → Jane Doe (manager relationship)

### **Test 2: HubSpot Format**  
1. Upload `hubspot_sample.csv`
2. **Expected**: Detects as "hubspot", creates 6 users
3. **Fields**: Employee Name → Name, Work Email → Email

### **Test 3: Workday/Zoho Format**
1. Upload `workday_sample.csv` or `zoho_sample.csv`
2. **Expected**: May detect as "hubspot" (same format now)
3. **Result**: Still works perfectly - creates users correctly

### **Test 4: Fallback Behavior**
1. Stop Python service (Ctrl+C in Terminal 1)
2. Upload any CSV
3. **Expected**: 
   - Console shows "Hierarchy processor service unavailable"
   - Upload still works with original CSV format
   - Graceful fallback behavior

## 📊 **Expected Data Transformation**

### **Input (Workday/Zoho CSV):**
```csv
user_id,first_name,middle_name,last_name,email,active,org_name,business_title,super_ref,managername,position_title,division,city,state,country,hire_date,termination_date,category
WD001,John,Michael,Smith,john.smith@company.com,true,Sales Organization,Senior Account Executive,WD002,Jane Doe,Account Executive,North America Sales,New York,NY,USA,2023-01-15,,Full Time
```

### **Output (Node.js Expected Format):**
```javascript
{
  "Name": "John Michael Smith",
  "Email": "john.smith@company.com", 
  "Role Title": "Senior Account Executive",
  "Reporting Manager Name": "Jane Doe",
  "Department": "North America Sales",
  "Location": "New York, NY, USA",
  "Employee ID": "WD001",
  "Start Date": "2023-01-15",
  "Employment Status": "Active"
}
```

## 🐛 **Troubleshooting**

### **Python Service Issues:**
```bash
# Check if running
curl http://localhost:8000/health

# Check hierarchy processor
curl http://localhost:8000/api/hierarchy/systems

# Restart if needed
cd crenovent-ai-service
python main.py
```

### **Node.js Issues:**
```bash
# Check if running
curl http://localhost:3000/api/health

# Check logs for normalization attempts
# Look for: "CSV normalized successfully" or "Hierarchy processor service unavailable"
```

### **Frontend Issues:**
- **F12 → Network tab**: Check if `/user-signup-csv` API call succeeds
- **F12 → Console**: Look for JavaScript errors or success messages
- **Check user list**: Verify users were actually created

## ✅ **Success Criteria**

**✅ Integration Working When:**
1. All test CSV formats upload successfully
2. Users created with correct data mapping
3. Console shows normalization success messages  
4. Hierarchy relationships preserved
5. RBAC permissions assigned correctly
6. Email notifications sent (if configured)

**🎉 Ready for Production When:**
- All 4 test files work
- Fallback behavior works when Python service is down
- No JavaScript console errors
- Users can login with assigned permissions

## 📞 **Need Help?**

If any test fails:
1. **Check both services are running** (ports 8000 and 3000)
2. **Check browser console** for error messages
3. **Check server logs** in both terminals
4. **Try the HTML test interface** first: `crenovent-ai-service/frontend_test.html`

**The system is designed to work even if the Python service is down - it will just use the original CSV format without normalization.**
