# Hierarchy Processor - CSV Field Normalizer

A Python microservice that normalizes CSV exports from various HRMS/CRM systems (Salesforce, HubSpot, Zoho, etc.) to a standardized format for your Node.js backend.

## 🎯 Purpose

This service acts as a **field mapping layer** between different HRMS CSV formats and your existing Node.js backend logic. It ensures that regardless of the source system (Salesforce, HubSpot, Zoho, etc.), your backend always receives data in the expected format.

## 🏗️ Architecture

```
HRMS CSV → Python Service → Normalized Data → Your Node.js Backend
```

- **Input**: Any HRMS CSV format
- **Processing**: Dynamic field detection and mapping
- **Output**: Standardized format matching your Node.js expectations

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd crenovent-ai-service
pip install -r requirements_hierarchy.txt
```

### 2. Start the Service

```bash
cd crenovent-ai-service
python -m hierarchy_processor.main
```

The service will start on port 8001 to avoid conflicts with the main AI service.

### 3. Test the Service

```bash
curl http://localhost:8001/health
```

## 📡 API Endpoints

### Main Endpoint: `/normalize-csv`

**POST** - Normalizes CSV data to your Node.js backend format

```json
{
  "csv_data": [
    {
      "Full Name": "John Smith",
      "Email": "john@company.com", 
      "Manager Email": "manager@company.com",
      "Profile": "Sales Manager"
    }
  ],
  "tenant_id": "1300",
  "uploaded_by": 1323
}
```

**Response:**
```json
{
  "success": true,
  "detected_system": "salesforce",
  "confidence": 0.95,
  "normalized_data": [
    {
      "Name": "John Smith",
      "Email": "john@company.com",
      "Role Title": "Sales Manager",
      "Reporting Email": "manager@company.com",
      "Reporting Manager Name": "",
      "Region": "",
      "Territory": "",
      // ... all standard fields
    }
  ]
}
```

### Other Endpoints

- `GET /health` - Health check
- `GET /systems` - List available HRMS configurations
- `POST /upload-csv` - Upload CSV file directly
- `GET /detect-system?headers=...` - Detect system from headers

## 🔧 Integration with Node.js Backend

Add this simple function to your existing `handleUserSignupCSV`:

```javascript
// Add this function to your Node.js backend
async function normalizeCSVFields(jsonData) {
  try {
    const response = await fetch('http://localhost:8001/normalize-csv', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ csv_data: jsonData })
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log(`✅ CSV normalized from ${result.detected_system} format`);
      return result.normalized_data;
    } else {
      console.warn('⚠️ CSV normalization failed, using original data');
      return jsonData;
    }
  } catch (error) {
    console.warn('⚠️ Python service unavailable, using original CSV format');
    return jsonData; // Graceful fallback
  }
}

// In your existing handleUserSignupCSV function, add this line:
async function handleUserSignupCSV({ req, res, next, pool, JWT_SECRET }) {
  try {
    // Your existing ExcelJS parsing
    let jsonData = await parseExcelFile(req.file.buffer);
    
    // NEW: Add this one line
    jsonData = await normalizeCSVFields(jsonData);
    
    // Your existing logic continues unchanged
    const cleanedData = jsonData.map(row => {
      // ... your existing cleaning logic
    });
    
    // ... rest of your existing code
  } catch (error) {
    // Your existing error handling
  }
}
```

## 📋 Supported HRMS Systems

### Currently Configured:
- **Salesforce** - User exports with profiles, territories, manager hierarchies
- **HubSpot** - Contact exports with job titles, managers, regions  
- **Zoho People** - Employee exports with designations, reporting managers
- **Generic** - Fallback for unknown systems

### Adding New Systems:
1. Create new YAML file in `hierarchy_processor/config/mappings/newsystem.yaml`
2. Define field patterns and mappings
3. Restart the service - auto-detected!

## 🔍 How It Works

### 1. **System Detection**
- Analyzes CSV headers using fuzzy matching
- Compares against known HRMS patterns
- Returns confidence score for detection

### 2. **Field Mapping**
- Maps source fields to your standard format
- Uses configurable patterns and fuzzy matching
- Handles field name variations automatically

### 3. **Data Normalization**
- Converts data types and formats
- Applies system-specific transformations
- Validates output data quality

### 4. **Output Standardization**
Your Node.js backend always receives these fields:
- `Name` - User's full name
- `Email` - Work email address
- `Role Title` - Job title/role
- `Reporting Email` - Manager's email
- `Reporting Manager Name` - Manager's name
- `Region`, `Area`, `District`, `Territory`, `Segment` - Geographic hierarchy

## ⚙️ Configuration

All HRMS mappings are in `hierarchy_processor/config/mappings/` as YAML files:

```yaml
# Example: salesforce.yaml
fields:
  Name:
    patterns: ["Full Name", "Name", "Username"]
    required: true
    type: "string"
  
  Email:
    patterns: ["Email", "Email Address", "Work Email"]
    required: true
    type: "email"
  
  Role Title:
    patterns: ["Profile", "Role", "User Role"]
    required: false
    type: "string"

processing_rules:
  role_mapping:
    "System Administrator": "admin"
    "Sales Manager": "manager"
    "Sales User": "sales"
```

## 🛠️ Customization

### Add New HRMS System:
1. Create `hierarchy_processor/config/mappings/yoursystem.yaml`
2. Define field patterns and mappings
3. Add any special processing rules
4. Test with sample CSV

### Modify Existing Mappings:
1. Edit the relevant YAML file
2. Add new field patterns
3. Update processing rules as needed
4. Restart service to reload config

## 📊 Monitoring & Debugging

### Logs
The service provides detailed logging:
- System detection results
- Field mapping confidence scores
- Processing statistics
- Error details

### Validation
Built-in validation checks:
- CSV structure validation
- Field mapping validation  
- Data quality validation
- Hierarchy validation

### Health Checks
- `/health` - Service status
- `/systems` - Available configurations
- Error handling with graceful fallbacks

## 🔒 Production Considerations

1. **CORS Configuration** - Update allowed origins
2. **Error Handling** - Comprehensive error responses
3. **Performance** - Handles large CSVs efficiently
4. **Monitoring** - Built-in health checks and logging
5. **Scalability** - Stateless design for easy scaling

## 🧪 Testing

Test with sample data:
```bash
curl -X POST http://localhost:8001/normalize-csv \
  -H "Content-Type: application/json" \
  -d '{"csv_data": [{"Full Name": "Test User", "Email": "test@company.com"}]}'
```

Your service is now ready to handle any HRMS CSV format while keeping your Node.js backend logic unchanged!
