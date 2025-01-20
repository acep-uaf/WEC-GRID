# --- Helper Functions ---
function Write-Color {
    param (
        [string]$Message,
        [string]$Color = "White"
    )
    $Host.UI.RawUI.ForegroundColor = $Color
    Write-Output $Message
    $Host.UI.RawUI.ForegroundColor = "White"
}

function Verify-Command {
    param (
        [string]$CommandName,
        [string]$ErrorMessage
    )
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        Write-Color $ErrorMessage "Red"
        exit 1
    }
}

function Check-CondaEnvironmentExists {
    param (
        [string]$envName
    )

    # Get the list of Conda environments
    $condaEnvs = conda env list | ForEach-Object { $_.Trim() } | Where-Object { -not ($_ -match "^#") }

    # Check if the environment exists
    if ($condaEnvs -contains $envName) {
        Write-Color "Conda environment '$envName' already exists." "Yellow"
        return $true
    } else {
        Write-Color "Conda environment '$envName' does not exist." "Cyan"
        return $false
    }
}

# --- Step 1: Check and Create Conda Environment ---
function Setup-CondaEnvironment {
    Write-Color "Step 1: Setting up Conda environment from YML..." "Cyan"
    Verify-Command "conda" "'conda' is NOT installed or not found in PATH. Please install Miniconda or Anaconda."

    $envName = "WEC_GRID_ENV"
    $ymlFile = "wec_grid_env.yml"

    if (-not (Test-Path $ymlFile)) {
        Write-Color "YAML file '$ymlFile' not found. Cannot create Conda environment." "Red"
        exit 1
    }

    if (Check-CondaEnvironmentExists -envName $envName) {
        conda env create -f $ymlFile
        if (-not $?) {
            Write-Color "Failed to create Conda environment '$envName'." "Red"
            exit 1
        }
    } else { 
        Write-Color "Skipping environment creation since it already exists." "Yellow"
    }

    conda activate $envName

    Write-Color "Checking if Conda environment 'WEC_GRID_ENV' is active..." "Cyan"
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


    Write-Color "`n Validating installed packages from 'wec_grid_env.yml'..." "Cyan"
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
    Write-Color "Conda environment '$envName' is ready and activated." "Green"
}

# --- Step 2: Install WEC_GRID ---
function Install-WEC_GRID {
    Write-Color "`nStep 2: Installing WEC_GRID package..." "Cyan"

    if (-not (Test-Path "./setup.py")) {
        Write-Color "The 'setup.py' file for WEC_GRID is not found in the current directory." "Red"
        exit 1
    }

    python -m pip install -e .
    if (-not $?) {
        Write-Color "Failed to install WEC_GRID package." "Red"
        exit 1
    }

    Write-Color "`nStep 3: Verifying WEC-GRID is installed and accessible..." "Cyan"
    try {
        python -c "import wec_grid" 2>&1 | Out-Null
        Write-Color "'WEC-GRID' is installed and accessible." "Green"
    } catch {
        Write-Color "'WEC-GRID' is NOT installed or accessible." "Red"
        exit 1
    }


    Write-Color "WEC_GRID package installed successfully!" "Green"
}

# --- Step 3: Configure PSSE ---
function Configure-PSSE {
    Write-Color "`nStep 3: Configuring PSSE..." "Cyan"

    $DefaultPssePath = "C:\Program Files\PTI\PSSE35\35.3"
    if (-not $env:PSSE_PATH) {
        $env:PSSE_PATH = $DefaultPssePath
    }

    if (-not (Test-Path -Path $env:PSSE_PATH)) {
        Write-Color "Default PSSE path '$DefaultPssePath' does not exist." "Yellow"
        $env:PSSE_PATH = Read-Host "Enter the correct PSSE installation path"
    }

    if (-not (Test-Path -Path $env:PSSE_PATH)) {
        Write-Color "Invalid PSSE path: '$env:PSSE_PATH'. Please check and try again." "Red"
        exit 1
    }

    $pssePaths = @(
        "$env:PSSE_PATH\PSSPY37",
        "$env:PSSE_PATH\PSSBIN",
        "$env:PSSE_PATH\PSSLIB",
        "$env:PSSE_PATH\EXAMPLE"
    )

    # Add PSSE paths to $env:PATH
    foreach ($path in $pssePaths) {
        if (-not $env:PATH.Contains($path)) {
            $env:PATH += ";$path"
        }
    }

    # Ensure $env:PYTHONPATH is initialized
    if (-not $env:PYTHONPATH) {
        $env:PYTHONPATH = ""
    }

    # Add PSSE Python path to $env:PYTHONPATH
    $pythonPath = "$env:PSSE_PATH\PSSPY37"
    if (-not $env:PYTHONPATH.Contains($pythonPath)) {
        $env:PYTHONPATH += ";$pythonPath"
    }

    Write-Color "`n Verifying PSSe is configured and accessible..." "Cyan"
    try {
        python -c "import psse35; import psspy" 2>&1 | Out-Null
        Write-Color "PSSe is configured and accessible." "Green"
    } catch {
        Write-Color "PSSe is NOT configured or accessible." "Red"
        exit 1
    }

    Write-Color "PSSE paths configured successfully!" "Green"
}

# --- Step 4: Configure MATLAB ---
function Configure-MATLAB {
    Write-Color "`nStep 4: Configuring MATLAB Engine API..." "Cyan"
    $DefaultMatlabPath = "C:\Program Files\MATLAB\R2021b"
    if (-not $env:MATLAB_PATH) {
        $env:MATLAB_PATH = $DefaultMatlabPath
    }

    if (-not (Test-Path -Path $env:MATLAB_PATH)) {
        Write-Color "Default MATLAB path '$DefaultMatlabPath' does not exist." "Yellow"
        $env:MATLAB_PATH = Read-Host "Enter the correct MATLAB installation path"
    }

    $matlabEnginePath = "$env:MATLAB_PATH\extern\engines\python"
    if (-not (Test-Path -Path $matlabEnginePath)) {
        Write-Color "Invalid MATLAB Engine path: '$matlabEnginePath'. Please check and try again." "Red"
        exit 1
    }

    Write-Color "Installing MATLAB Engine API..." "Yellow"
    Push-Location $matlabEnginePath
    python -m pip install .
    Pop-Location

    $matlabPythonPath = "$matlabEnginePath"
    if (-not $env:PYTHONPATH.Contains($matlabPythonPath)) {
        $env:PYTHONPATH += ";$matlabPythonPath"
    }

    Write-Color "`n Verifying MATLAB Engine API is installed..." "Cyan"
    try {
        python -c "import matlab.engine" 2>&1 | Out-Null
        Write-Color "'matlab.engine' is installed and accessible." "Green"
    } catch {
        Write-Color "'matlab.engine' is NOT installed or accessible." "Red"
        exit 1
    }

    Write-Color "MATLAB paths configured successfully!" "Green"
}

# --- Step 5: Verify Environment ---


# --- Main Script Execution ---
Setup-CondaEnvironment
Install-WEC_GRID
Configure-PSSE
Configure-MATLAB
Write-Color "`nSetup complete! Your environment is ready." "Yellow"