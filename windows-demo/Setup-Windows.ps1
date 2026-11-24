$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    $versionCheck = "import struct,sys; sys.exit(not (sys.version_info[:2] == (3,10) and struct.calcsize('P') == 8))"
    if (Get-Command python.exe -ErrorAction SilentlyContinue) {
        python -c $versionCheck
        $pathPythonOk = $LASTEXITCODE -eq 0
    } else {
        $pathPythonOk = $false
    }
    if ($pathPythonOk) {
        python -m venv .venv
    } elseif (Get-Command py.exe -ErrorAction SilentlyContinue) {
        py -3.10 -m venv .venv
    } else {
        throw 'Install 64-bit Python 3.10 and add it to PATH.'
    }
    if ($LASTEXITCODE) { throw 'Python virtual environment setup failed' }
}
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $python -c "import struct,sys; assert sys.version_info[:2] == (3,10) and struct.calcsize('P') == 8, 'Use 64-bit Python 3.10'"
if ($LASTEXITCODE) { throw 'Python 3.10 x64 required' }
& $python -m pip install -r requirements-windows.txt
if ($LASTEXITCODE) { throw 'Dependency installation failed' }
& $python verify_assets.py
if ($LASTEXITCODE) { throw 'Model verification failed. Fetch the actual model files before starting.' }

