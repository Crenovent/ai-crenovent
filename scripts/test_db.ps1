# Quick Database Test Script for Windows
# Usage: .\test_db.ps1

Write-Host "🔍 Testing Foundation Database..." -ForegroundColor Cyan
Write-Host "=" * 40

# Check if DATABASE_URL is set
if (-not $env:DATABASE_URL) {
    Write-Host "❌ DATABASE_URL environment variable not set" -ForegroundColor Red
    Write-Host "💡 Set it like: `$env:DATABASE_URL='postgresql://user:pass@host:port/db'" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ DATABASE_URL found" -ForegroundColor Green

# Test Python and asyncpg
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found" -ForegroundColor Red
    exit 1
}

try {
    python -c "import asyncpg; print('✅ asyncpg module available')" 2>$null
    Write-Host "✅ asyncpg module available" -ForegroundColor Green
} catch {
    Write-Host "⚠️ asyncpg not available, installing..." -ForegroundColor Yellow
    pip install asyncpg
}

Write-Host ""
Write-Host "🚀 Running database status check..." -ForegroundColor Cyan

# Run the status check
python check_db_status.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "🎉 Database test completed successfully!" -ForegroundColor Green
    Write-Host "📋 Next steps:" -ForegroundColor Cyan
    Write-Host "   • Run full verification: python verify_foundation.py" -ForegroundColor White
    Write-Host "   • Deploy RBA agents" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "🔧 Database needs attention" -ForegroundColor Yellow
    Write-Host "   • Check DATABASE_URL" -ForegroundColor White
    Write-Host "   • Run FIXED_AZURE_SCHEMA.sql" -ForegroundColor White
}
