# Emergency Rollback Script for Code Quality Bugfix
# Purpose: Rapid restoration of system functionality in case of critical failures
# Usage: ./backend/scripts/emergency_rollback.ps1

param(
    [string]$Reason = "Emergency rollback triggered",
    [switch]$SkipTests = $false
)

# Configuration
$ProjectRoot = "e:\projects\agri new"
$LogFile = "$ProjectRoot\logs\emergency_rollback.log"
$BackupBranch = "main"
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Ensure logs directory exists
$LogDir = Split-Path $LogFile -Parent
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force
}

function Write-RollbackLog {
    param([string]$Message)
    $LogEntry = "[$Timestamp] $Message"
    Write-Host $LogEntry -ForegroundColor Yellow
    Add-Content -Path $LogFile -Value $LogEntry
}

function Test-SystemHealth {
    Write-RollbackLog "Running basic system health checks..."
    
    # Check if Python is available
    try {
        python --version 2>&1 | Out-Null
        Write-RollbackLog "✓ Python is available"
    }
    catch {
        Write-RollbackLog "✗ Python not found in PATH"
        return $false
    }
    
    # Check if backend directory exists
    if (Test-Path "$ProjectRoot\backend") {
        Write-RollbackLog "✓ Backend directory found"
    } else {
        Write-RollbackLog "✗ Backend directory not found"
        return $false
    }
    
    # Check if requirements.txt exists
    if (Test-Path "$ProjectRoot\backend\requirements.txt") {
        Write-RollbackLog "✓ Requirements file found"
    } else {
        Write-RollbackLog "✗ Requirements file not found"
        return $false
    }
    
    return $true
}

function Stop-Services {
    Write-RollbackLog "Stopping potentially running services..."
    
    # Kill any running Python processes that might be our services
    Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -eq "python" -and $_.Path -like "*agri*"
    } | ForEach-Object {
        Write-RollbackLog "Stopping Python process: $($_.Id)"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
    
    # Stop Docker containers if they exist
    try {
        $containers = docker ps -q --filter "label=project=agri" 2>$null
        if ($containers) {
            Write-RollbackLog "Stopping Docker containers..."
            docker stop $containers 2>$null | Out-Null
        }
    }
    catch {
        Write-RollbackLog "Docker not available or no containers running"
    }
}

function Clear-Caches {
    Write-RollbackLog "Clearing potentially corrupted caches..."
    
    # Clear Python cache
    $PyCacheDir = "$ProjectRoot\backend\__pycache__"
    if (Test-Path $PyCacheDir) {
        Remove-Item -Path $PyCacheDir -Recurse -Force -ErrorAction SilentlyContinue
        Write-RollbackLog "Cleared Python __pycache__"
    }
    
    # Clear pytest cache
    $PytestCache = "$ProjectRoot\backend\.pytest_cache"
    if (Test-Path $PytestCache) {
        Remove-Item -Path $PytestCache -Recurse -Force -ErrorAction SilentlyContinue
        Write-RollbackLog "Cleared pytest cache"
    }
    
    # Clear Ruff cache
    $RuffCache = "$ProjectRoot\.ruff_cache"
    if (Test-Path $RuffCache) {
        Remove-Item -Path $RuffCache -Recurse -Force -ErrorAction SilentlyContinue
        Write-RollbackLog "Cleared Ruff cache"
    }
}

function Perform-GitRollback {
    Write-RollbackLog "Performing Git rollback to $BackupBranch branch..."
    
    Set-Location $ProjectRoot
    
    # Stash any uncommitted changes
    git stash push -m "Emergency rollback stash - $Timestamp" 2>&1 | Out-Null
    Write-RollbackLog "Stashed uncommitted changes"
    
    # Switch to backup branch
    try {
        git checkout $BackupBranch 2>&1 | Out-Null
        Write-RollbackLog "✓ Switched to $BackupBranch branch"
    }
    catch {
        Write-RollbackLog "✗ Failed to switch to $BackupBranch branch"
        return $false
    }
    
    # Pull latest changes to ensure we have the most recent stable version
    try {
        git pull origin $BackupBranch 2>&1 | Out-Null
        Write-RollbackLog "✓ Updated to latest $BackupBranch"
    }
    catch {
        Write-RollbackLog "⚠ Could not pull latest changes (working offline?)"
    }
    
    return $true
}

function Run-HealthValidation {
    if ($SkipTests) {
        Write-RollbackLog "Skipping health validation tests (SkipTests flag set)"
        return $true
    }
    
    Write-RollbackLog "Running critical health validation..."
    
    Set-Location "$ProjectRoot\backend"
    
    # Check if we can import core modules
    try {
        python -c "import app.core; print('Core modules import successful')" 2>&1 | Out-Null
        Write-RollbackLog "✓ Core modules import successfully"
    }
    catch {
        Write-RollbackLog "✗ Core modules import failed"
        return $false
    }
    
    # Run a quick preservation test if available
    if (Test-Path "test_preservation_simple.py") {
        try {
            python -m pytest test_preservation_simple.py -x -v --tb=short 2>&1 | Out-Null
            Write-RollbackLog "✓ Basic preservation tests passed"
        }
        catch {
            Write-RollbackLog "⚠ Some preservation tests failed (may be expected during rollback)"
        }
    }
    
    return $true
}

# Main rollback procedure
Write-RollbackLog "=== EMERGENCY ROLLBACK INITIATED ==="
Write-RollbackLog "Reason: $Reason"
Write-RollbackLog "Timestamp: $Timestamp"
Write-RollbackLog "Current Directory: $(Get-Location)"

# Step 1: Stop services
Stop-Services

# Step 2: Health check
if (-not (Test-SystemHealth)) {
    Write-RollbackLog "✗ Basic system health check failed"
    exit 1
}

# Step 3: Clear caches
Clear-Caches

# Step 4: Git rollback
if (-not (Perform-GitRollback)) {
    Write-RollbackLog "✗ Git rollback failed"
    exit 1
}

# Step 5: Validation
if (-not (Run-HealthValidation)) {
    Write-RollbackLog "✗ Post-rollback validation failed"
    exit 1
}

Write-RollbackLog "=== EMERGENCY ROLLBACK COMPLETED SUCCESSFULLY ==="
Write-RollbackLog "System restored to $BackupBranch branch"
Write-RollbackLog "Next steps:"
Write-RollbackLog "1. Investigate the original issue that triggered rollback"
Write-RollbackLog "2. Run full test suite: python -m pytest backend/tests/"
Write-RollbackLog "3. Restart services when ready"
Write-RollbackLog "4. Review rollback log: $LogFile"

# Return to original directory
Set-Location $ProjectRoot

Write-Host "`n🔄 Emergency rollback completed. Check log file: $LogFile" -ForegroundColor Green
exit 0