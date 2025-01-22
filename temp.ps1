# Define the test environment's prefix
$env:CONDA_PREFIX = "C:\Users\alexb\miniconda3\envs\test"

# Define the global Conda environment name
$Global:CondaEnvName = "test"

# Define the Configure-PSSE-Persistent function
function Configure-PSSE-Persistent {
    param (
        [string]$PsseBasePath = "C:\Program Files\PTI\PSSE35\35.3"
    )

    # Validate Conda environment name
    if (-not $Global:CondaEnvName) {
        Write-Output "Error: Conda environment name is not defined."
        return
    }

    Write-Output "Configuring PSSE environment variables for Conda environment '$Global:CondaEnvName'..."

    # Define the paths
    $pssePaths = @(
        "$PsseBasePath\PSSPY37",
        "$PsseBasePath\PSSBIN",
        "$PsseBasePath\PSSLIB",
        "$PsseBasePath\EXAMPLE"
    )

    # Directories for activate.d and deactivate.d
    $activateDir = "$env:CONDA_PREFIX\etc\conda\activate.d"
    $deactivateDir = "$env:CONDA_PREFIX\etc\conda\deactivate.d"

    # Ensure directories exist
    if (-not (Test-Path $activateDir)) { New-Item -ItemType Directory -Path $activateDir -Force }
    if (-not (Test-Path $deactivateDir)) { New-Item -ItemType Directory -Path $deactivateDir -Force }

    # Write activate.ps1
    $activatePs1 = @"
# Add PSSE paths only if not already set
if (-not `$Env:PSSE_PATHS_SET) {
    `$Env:PATH += ";$($pssePaths -join ";")"
    `$Env:PYTHONPATH = "$PsseBasePath\PSSPY37"
    `$Env:PSSE_PATHS_SET = "1"
}
"@
    $activateFile = "$activateDir\set_psse_paths.ps1"
    Set-Content -Path $activateFile -Value $activatePs1
    Write-Output "Created: $activateFile"

    # Write deactivate.ps1
    $deactivatePs1 = @"
# Remove PSSE paths from PATH and clear PYTHONPATH
if (`$Env:PSSE_PATHS_SET) {
    `$Env:PATH = `$Env:PATH -replace ";$($pssePaths -join ";")", ""
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PSSE_PATHS_SET -ErrorAction SilentlyContinue
}
"@
    $deactivateFile = "$deactivateDir\unset_psse_paths.ps1"
    Set-Content -Path $deactivateFile -Value $deactivatePs1
    Write-Output "Created: $deactivateFile"

    Write-Output "PSSE environment configuration is complete for Conda environment '$Global:CondaEnvName'."
    Write-Output "Activate the environment to apply the changes."
}

# Call the function
Configure-PSSE-Persistent