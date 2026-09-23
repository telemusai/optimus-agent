# Install/update the newest stable version tag. Builds native Windows Rust locally.
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'Use install.sh on Linux/macOS.' }
foreach ($command in @('git', 'cargo', 'uv')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Missing $command. See https://github.com/telemusai/optimus-agent#install-and-update for prerequisites."
    }
}
$python = Get-Command py -ErrorAction SilentlyContinue
$pythonArgs = @('-3')
if (-not $python) {
    $python = Get-Command python -ErrorAction SilentlyContinue
    $pythonArgs = @()
}
if (-not $python) { throw 'Install Python 3.11+ and reopen your terminal.' }
& $python.Source @pythonArgs -c 'import sys; sys.exit(sys.version_info < (3, 11))'
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ is required.' }
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('optimus-install-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $script = Join-Path $temporary 'install.py'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://telemus.ai/optimus-agent/install.py' -OutFile $script
    & $python.Source @pythonArgs $script
    if ($LASTEXITCODE -ne 0) { throw 'Optimus installation failed; see the error above.' }
    $bin = Join-Path $env:USERPROFILE '.local\bin'
    $userPath = [string][Environment]::GetEnvironmentVariable('Path', 'User')
    if (($userPath -split ';') -notcontains $bin) {
        [Environment]::SetEnvironmentVariable('Path', (($userPath.TrimEnd(';') + ';' + $bin).TrimStart(';')), 'User')
    }
    $env:Path = "$bin;$env:Path"
    Write-Host 'Open a new terminal and run optimus-agent from your project directory.'
} finally {
    Remove-Item -LiteralPath $temporary -Recurse -Force
}
