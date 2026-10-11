param([switch]$SkipDependencies)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
foreach ($program in @('git','py','node','npm.cmd')) {
    if (-not (Get-Command $program -ErrorAction SilentlyContinue)) { throw "Missing program: $program; see README." }
}
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Please install Python 3.12 first.' }
}
$reviewPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not $SkipDependencies) {
    & $reviewPython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
    & $reviewPython -m pip install -r requirements.lock
    if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed.' }
    & $reviewPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed.' }
}
& $reviewPython -c "import sys,torch; assert sys.version_info[:2]==(3,12), 'Requires Python 3.12'; assert torch.cuda.is_available(), 'This reviewed setup requires a working NVIDIA CUDA GPU'; print(torch.__version__,torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'Requires Python 3.12 and a working NVIDIA GPU. CPU runtime is not verified.' }
New-Item -ItemType Directory -Force -Path third_party | Out-Null
$repositories = @(
    @{Name='UniversalFakeDetect'; Url='https://github.com/WisconsinAIVision/UniversalFakeDetect'; Commit='030495aea3300a8b54c0ec37ec7fe1dd7e63c619'},
    @{Name='TruFor'; Url='https://github.com/grip-unina/TruFor'; Commit='ae54475df6f41a491d7615100feb19263dec13f7'},
    @{Name='D3'; Url='https://github.com/Zig-HS/D3'; Commit='c798fbc57fe0c4198d63a73732c2c0f9e4b4816c'},
    @{Name='WaveRep'; Url='https://github.com/grip-unina/WaveRep-SyntheticVideoDetection'; Commit='0fd6010759c14b572b7842a28fa9f85fe1ddd2fd'}
)
foreach ($repository in $repositories) {
    $repositoryPath = Join-Path 'third_party' $repository.Name
    if (-not (Test-Path -LiteralPath $repositoryPath)) {
        git clone $repository.Url $repositoryPath
        if ($LASTEXITCODE -ne 0) { throw "Upstream clone failed: $($repository.Name)" }
    }
    $changes = git -C $repositoryPath status --porcelain
    if ($LASTEXITCODE -ne 0 -or $changes) { throw "Upstream directory is modified or not a Git repository: $repositoryPath" }
    git -C $repositoryPath checkout --detach $repository.Commit
    if ($LASTEXITCODE -ne 0) { throw "Pinned upstream commit unavailable: $($repository.Name)" }
}
& $reviewPython -X utf8 scripts/fetch_models.py
if ($LASTEXITCODE -ne 0) { throw 'Upstream download failed. Retry to reuse complete downloads.' }
& $reviewPython -X utf8 scripts/verify_assets.py
if ($LASTEXITCODE -ne 0) { throw 'Asset validation failed. Do not start the demo.' }
Push-Location frontend
try {
    npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
& $reviewPython -X utf8 scripts/make_demo_samples.py
if ($LASTEXITCODE -ne 0) { throw 'Procedural engineering input generation failed.' }
Write-Host 'Setup complete. Use the CMD launcher in the project root and keep it running.'
