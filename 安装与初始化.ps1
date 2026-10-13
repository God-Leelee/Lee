param([switch]$SkipDependencies, [switch]$RebuildFrontend, [string]$Wheelhouse = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Install Python 3.12 with the Windows py launcher first.' }
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 virtual environment creation failed.' }
}
$reviewPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not $SkipDependencies) {
    if ($Wheelhouse) {
        & $reviewPython -m pip install --no-index --find-links $Wheelhouse -r requirements.lock
    } else {
        & $reviewPython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
        if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
        & $reviewPython -m pip install -r requirements.lock
    }
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $reviewPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Pinned dependency conflict.' }
}
& $reviewPython -c "import sys,torch; assert sys.version_info[:2]==(3,12), 'Requires Python 3.12'; assert torch.cuda.is_available(), 'Requires NVIDIA CUDA GPU'; print(torch.__version__,torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 and a compatible NVIDIA driver are required.' }
$assetLock = Get-Content -LiteralPath 'models/runtime_assets.json' -Raw | ConvertFrom-Json
foreach ($repo in $assetLock.repositories) {
    $repoPath = Join-Path 'third_party' $repo.name
    if (-not (Test-Path -LiteralPath $repoPath)) {
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Extract both model archives, or install Git to fetch upstream source.' }
        git clone $repo.source $repoPath
        if ($LASTEXITCODE -ne 0) { throw 'Pinned upstream source download failed.' }
        git -C $repoPath checkout --detach $repo.commit
        if ($LASTEXITCODE -ne 0) { throw 'Pinned upstream commit unavailable.' }
    }
}
& $reviewPython -X utf8 scripts/fetch_models.py
if ($LASTEXITCODE -ne 0) { throw 'Asset download/hash check failed. Existing valid assets are reusable.' }
& $reviewPython -X utf8 scripts/verify_assets.py
if ($LASTEXITCODE -ne 0) { throw 'Runtime asset lock verification failed.' }
if ($RebuildFrontend -or -not (Test-Path -LiteralPath 'frontend/dist/index.html')) {
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'Install Node.js 20+ to rebuild the frontend.' }
    Push-Location frontend
    try {
        npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
        npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend compilation failed.' }
    } finally { Pop-Location }
}
& $reviewPython -X utf8 scripts/make_demo_samples.py
if ($LASTEXITCODE -ne 0) { throw 'Demo input generation failed.' }
Write-Host 'CLEARTRACE 1.8 ready. Double-click the CMD launcher, wait for model startup, then open http://127.0.0.1:8765/'
