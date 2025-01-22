# Function to display messages with colors
function Write-Color {
    param (
        [string]$Message,
        [string]$Color = "White"
    )
    $Host.UI.RawUI.ForegroundColor = $Color
    Write-Output $Message
    $Host.UI.RawUI.ForegroundColor = "White"
}

# Step 1: Confirm active Conda environment is WEC_GRID_ENV
Write-Color "Step 1: Checking if Conda environment 'WEC_GRID_ENV' is active..." "Cyan"
if (Get-Command conda -ErrorAction SilentlyContinue) {
    $activeEnv = conda info --json | ConvertFrom-Json | Select-Object -ExpandProperty active_prefix
    if ($activeEnv -match "WEC_GRID_ENV") {
        Write-Color "The active Conda environment is 'WEC_GRID_ENV'." "Green"
    } else {
        Write-Color "The active Conda environment is NOT 'WEC_GRID_ENV'." "Red"
        Write-Color "Please activate the 'WEC_GRID_ENV' environment and try again." "Yellow"
        exit 1
    }
} else {
    Write-Color "'conda' is NOT installed or not found in PATH." "Red"
    exit 1
}

# Step 2: Verify Python version is 3.7
Write-Color "`nStep 2: Verifying Python version is 3.7..." "Cyan"
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "3.7") {
        Write-Color "Python version is 3.7." "Green"
    } else {
        Write-Color "Python version is NOT 3.7. Detected: $pythonVersion" "Red"
        exit 1
    }
} else {
    Write-Color "Python is NOT installed or not found in PATH." "Red"
    exit 1
}

# Step 3: Check if WEC-GRID is installed and accessible
Write-Color "`nStep 3: Checking if WEC-GRID is installed and accessible..." "Cyan"
try {
    python -c "import wec_grid" 2>&1 | Out-Null
    Write-Color "'WEC-GRID' is installed and accessible." "Green"
} catch {
    Write-Color "'WEC-GRID' is NOT installed or accessible." "Red"
    exit 1
}

# Step 4: Validate packages from wec_grid_env.yml
Write-Color "`nStep 4: Validating installed packages from 'wec_grid_env.yml'..." "Cyan"
$ymlPath = "./wec_grid_env.yml"
if (Test-Path $ymlPath) {
    $ymlContent = Get-Content $ymlPath -Raw
    $dependencies = ($ymlContent -split "`n") | Where-Object { $_ -match "^- " } | ForEach-Object { ($_ -replace "^- ", "").Trim() }

    $missingPackages = @()
    foreach ($pkg in $dependencies) {
        $pkgName = if ($pkg -match "=") { $pkg.Split("=")[0] } else { $pkg }
        try {
            conda list | Select-String -Quiet $pkgName | Out-Null
        } catch {
            $missingPackages += $pkgName
        }
    }

    if ($missingPackages.Count -eq 0) {
        Write-Color "All packages specified in 'wec_grid_env.yml' are installed." "Green"
    } else {
        Write-Color "`nThe following packages are NOT installed:" "Red"
        $missingPackages | ForEach-Object { Write-Color "- $_" "Yellow" }
    }
} else {
    Write-Color "'wec_grid_env.yml' file not found at $ymlPath." "Red"
    exit 1
}

# Step 5: Validate pip packages from requirements.txt
Write-Color "`nStep 5: Validating pip packages from 'requirements.txt'..." "Cyan"
$reqPath = "./requirements.txt"
if (Test-Path $reqPath) {
    $reqPackages = Get-Content $reqPath
    $missingPipPackages = @()
    foreach ($pkg in $reqPackages) {
        try {
            python -m pip show $pkg 2>&1 | Out-Null
        } catch {
            $missingPipPackages += $pkg
        }
    }

    if ($missingPipPackages.Count -eq 0) {
        Write-Color "All pip packages specified in 'requirements.txt' are installed." "Green"
    } else {
        Write-Color "`nThe following pip packages are NOT installed:" "Red"
        $missingPipPackages | ForEach-Object { Write-Color "- $_" "Yellow" }
    }
} else {
    Write-Color "'requirements.txt' file not found at $reqPath." "Red"
}

# Step 6: Check if MATLAB Engine API is installed
Write-Color "`nStep 6: Checking if MATLAB Engine API is installed..." "Cyan"
try {
    python -c "import matlab.engine" 2>&1 | Out-Null
    Write-Color "'matlab.engine' is installed and accessible." "Green"
} catch {
    Write-Color "'matlab.engine' is NOT installed or accessible." "Red"
    exit 1
}

# Step 7: Check if PSSe is configured and accessible
Write-Color "`nStep 7: Checking if PSSe is configured and accessible..." "Cyan"
try {
    python -c "import psse35; import psspy" 2>&1 | Out-Null
    Write-Color "PSSe is configured and accessible." "Green"
} catch {
    Write-Color "PSSe is NOT configured or accessible." "Red"
    exit 1
}

Write-Color "`nEnvironment setup verification completed successfully!" "Yellow"