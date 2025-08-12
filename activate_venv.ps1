# PowerShell script to activate the Claimsure virtual environment

Write-Host "🚀 Activating Claimsure Virtual Environment..." -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Cyan

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    # Activate the virtual environment
    & "venv\Scripts\Activate.ps1"
    
    Write-Host "✅ Virtual environment activated successfully!" -ForegroundColor Green
    Write-Host "📦 Python packages are now isolated in the venv directory" -ForegroundColor Yellow
    Write-Host "🔧 You can now run the Claimsure application" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "💡 Next steps:" -ForegroundColor Cyan
    Write-Host "   1. Set up your .env file with API keys" -ForegroundColor White
    Write-Host "   2. Start the API: python -m uvicorn src.api:app --host 0.0.0.0 --port 8000" -ForegroundColor White
    Write-Host "   3. Start the UI: python run_ui.py" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv venv" -ForegroundColor Yellow
    Write-Host "Then run this script again." -ForegroundColor Yellow
}
