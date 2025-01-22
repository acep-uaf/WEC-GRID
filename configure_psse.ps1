# Default PSSE and MATLAB installation paths
$DefaultPssePath = "C:\Program Files\PTI\PSSE35\35.3"
$DefaultMatlabPath = "C:\Program Files\MATLAB\R2021b"

# Save the current working directory
$OriginalDir = Get-Location

# --- Configure PSSE ---
Write-Host "Configuring PSSE..."
if (-not $env:PSSE_PATH) {
    $env:PSSE_PATH = $DefaultPssePath
}

if (-not (Test-Path -Path $env:PSSE_PATH)) {
    Write-Host "The default PSSE path '$DefaultPssePath' does not exist."
    $env:PSSE_PATH = Read-Host "Please enter the correct PSSE installation path (e.g., C:\Program Files\PTI\PSSE35\35.3)"
}

if (-not (Test-Path -Path $env:PSSE_PATH)) {
    Write-Host "The specified PSSE path '$env:PSSE_PATH' does not exist. Please check and try again."
    exit 1
}

$pssePaths = @(
    "$env:PSSE_PATH\PSSPY37",
    "$env:PSSE_PATH\PSSBIN",
    "$env:PSSE_PATH\PSSLIB",
    "$env:PSSE_PATH\EXAMPLE"
)

foreach ($path in $pssePaths) {
    if (-not $env:PATH.Contains($path)) {
        $env:PATH += ";$path"
    }
}

if (-not $env:PYTHONPATH) {
    $env:PYTHONPATH = ""
}

$pythonPath = "$env:PSSE_PATH\PSSPY37"
if ($env:PYTHONPATH -notcontains $pythonPath) {
    $env:PYTHONPATH += ";$pythonPath"
}

Write-Host "PSSE paths configured successfully!"
Write-Host "PSSE_PATH: $env:PSSE_PATH"

# --- Configure MATLAB ---
Write-Host "Configuring MATLAB Engine API..."
if (-not $env:MATLAB_PATH) {
    $env:MATLAB_PATH = $DefaultMatlabPath
}

if (-not (Test-Path -Path $env:MATLAB_PATH)) {
    Write-Host "The default MATLAB path '$DefaultMatlabPath' does not exist."
    $env:MATLAB_PATH = Read-Host "Please enter the correct MATLAB installation path (e.g., C:\Program Files\MATLAB\R2021b)"
}

$matlabEnginePath = "$env:MATLAB_PATH\extern\engines\python"

if (-not (Test-Path -Path $matlabEnginePath)) {
    Write-Host "The specified MATLAB path '$env:MATLAB_PATH' does not contain the MATLAB Engine API directory. Please check and try again."
    exit 1
}

# Install the MATLAB Engine API
Write-Host "Installing MATLAB Engine API..."
cd $matlabEnginePath
python -m pip install .

if (-not $env:PYTHONPATH) {
    $env:PYTHONPATH = ""
}

$matlabPythonPath = "$matlabEnginePath"
if ($env:PYTHONPATH -notcontains $matlabPythonPath) {
    $env:PYTHONPATH += ";$matlabPythonPath"
}

Write-Host "MATLAB paths configured successfully!"
Write-Host "MATLAB_PATH: $env:MATLAB_PATH"

cd $OriginalDir
# --- Summary ---
Write-Host "`nConfiguration Complete:"
Write-Host "PSSE_PATH: $env:PSSE_PATH"
Write-Host "MATLAB_PATH: $env:MATLAB_PATH"
Write-Host "PYTHONPATH: $env:PYTHONPATH"