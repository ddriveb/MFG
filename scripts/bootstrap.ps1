<# Create an independent Python 3.10 environment; never use SRP's interpreter. #>
param([string]$PythonExe = '')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot '.venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if ($PythonExe) {
            & $PythonExe -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"'
            if ($LASTEXITCODE -ne 0) { throw 'Selected Python is not 3.10.' }
            & $PythonExe -m venv $venvPath
        } else {
            & py -3.10 -m venv $venvPath
        }
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 environment creation failed.' }
    }
    & $venvPython -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"'
    if ($LASTEXITCODE -ne 0) { throw 'Existing .venv uses an incompatible Python.' }
    # A local .pth registers this clone without network access or package builds.
    $sitePackages = (& $venvPython -c 'import site; print(site.getsitepackages()[0])').Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot locate environment site-packages.' }
    [System.IO.File]::WriteAllText(
        (Join-Path $sitePackages 'mfg_hedge_local.pth'),
        "$(Join-Path $projectRoot 'src')`r`n", (New-Object System.Text.UTF8Encoding($false)))
    # Seven small frozen candidate inputs are needed by existing regression tests.
    # Restoration checks exact hashes and never overwrites differing local results.
    & $venvPython .\scripts\restore_frozen_inputs.py
    if ($LASTEXITCODE -ne 0) { throw 'Frozen runtime input verification failed.' }
    & $venvPython -m unittest discover -s .\tests -v
    if ($LASTEXITCODE -ne 0) { throw 'MFG test suite failed.' }
    & $venvPython -m mfg_hedge check --config .\configs\v1_minimal.json
    if ($LASTEXITCODE -ne 0) { throw 'MFG configuration check failed.' }
} finally {
    Pop-Location
}
