# setup.ps1
# ----------------------------------------------------------------------------
# Lexical Consensus — Initial Repository Setup for Windows
# ----------------------------------------------------------------------------
# Run this script in PowerShell from the directory where you want the
# repository to live. It will create the full directory structure and
# placeholder files.
#
# Usage:
#   1. Open PowerShell
#   2. Navigate to your projects folder (e.g. cd C:\Users\Patricio\dev)
#   3. Run: .\setup.ps1
#
# Note: If you get a script execution error, run this first:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# ----------------------------------------------------------------------------

$ProjectName = "lexical-consensus"
$Root = Join-Path (Get-Location) $ProjectName

Write-Host ""
Write-Host "Creating lexical-consensus repository structure..." -ForegroundColor Cyan
Write-Host "Location: $Root" -ForegroundColor Gray
Write-Host ""

# Create root directory
if (Test-Path $Root) {
    Write-Host "Directory $ProjectName already exists. Aborting." -ForegroundColor Red
    exit 1
}
New-Item -ItemType Directory -Path $Root | Out-Null

# Directory structure
$Directories = @(
    "docs",
    "src",
    "src\agents",
    "src\consensus",
    "src\dataset",
    "src\metrics",
    "src\utils",
    "src\graph",
    "experiments",
    "experiments\exp_001_baseline",
    "experiments\exp_002_control_conditions",
    "tests",
    "notebooks",
    "results",
    "scripts"
)

foreach ($dir in $Directories) {
    $fullPath = Join-Path $Root $dir
    New-Item -ItemType Directory -Path $fullPath | Out-Null
    Write-Host "  [+] $dir" -ForegroundColor Green
}

# Create __init__.py files for Python packages
$InitFiles = @(
    "src\__init__.py",
    "src\agents\__init__.py",
    "src\consensus\__init__.py",
    "src\dataset\__init__.py",
    "src\metrics\__init__.py",
    "src\utils\__init__.py",
    "src\graph\__init__.py",
    "tests\__init__.py"
)

foreach ($file in $InitFiles) {
    $fullPath = Join-Path $Root $file
    New-Item -ItemType File -Path $fullPath | Out-Null
}

# Create .gitkeep for empty dirs
New-Item -ItemType File -Path (Join-Path $Root "results\.gitkeep") | Out-Null

Write-Host ""
Write-Host "Structure created successfully." -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. cd $ProjectName"
Write-Host "  2. Copy the source files from the Claude session into their directories"
Write-Host "  3. git init"
Write-Host "  4. git add ."
Write-Host "  5. git commit -m 'Initial structure'"
Write-Host "  6. Open with Claude Code: 'claude' in this directory"
Write-Host ""
