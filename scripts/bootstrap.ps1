[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Stop-Bootstrap {
    param([string]$Code, [string]$Message)
    throw "$Code $Message"
}

function Get-Lock {
    param([string]$Path, [string]$Platform, [string]$Architecture)

    $scalars = @{}
    $assets = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -notmatch "=") {
            Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock contains a malformed record."
        }
        $key, $value = $line -split "=", 2
        switch ($key) {
            { $_ -in @("schema", "uv_version", "cpython_version") } {
                if ($scalars.ContainsKey($key)) {
                    Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock duplicates $key."
                }
                $scalars[$key] = $value
            }
            "asset" {
                $parts = $value -split "\|", 4
                if ($parts.Count -ne 4) {
                    Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock contains a malformed asset record."
                }
                $recordPlatform, $recordArchitecture, $filename, $sha256 = $parts
                if ($filename -match "[\\/]" -or $filename -in @("", ".", "..") -or $filename.StartsWith(".")) {
                    Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock contains an unsafe asset filename."
                }
                if ($sha256 -notmatch "^[0-9a-fA-F]{64}$") {
                    Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock contains an invalid SHA-256."
                }
                $recordKey = "$recordPlatform|$recordArchitecture"
                if ($assets.ContainsKey($recordKey)) {
                    Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock duplicates $recordKey."
                }
                if ($recordKey -notin @(
                    "windows|x86_64", "windows|aarch64", "macos|x86_64",
                    "macos|aarch64", "linux-gnu|x86_64", "linux-gnu|aarch64"
                )) {
                    Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock contains an unsupported platform record."
                }
                if ((($recordPlatform -eq "windows") -and -not $filename.EndsWith(".zip")) -or
                    (($recordPlatform -ne "windows") -and -not $filename.EndsWith(".tar.gz"))) {
                    Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock contains an invalid archive format."
                }
                $assets[$recordKey] = [pscustomobject]@{ Filename = $filename; Sha256 = $sha256 }
            }
            default { Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock contains an unknown key: $key" }
        }
    }
    if ($scalars.Keys.Count -ne 3 -or $scalars["schema"] -ne "1") {
        Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock must contain schema=1 exactly once."
    }
    if ($scalars["uv_version"] -ne "0.12.1" -or $scalars["cpython_version"] -ne "3.12.13") {
        Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock must pin uv 0.12.1 and CPython 3.12.13."
    }
    if ($scalars["uv_version"] -notmatch "^\d+\.\d+\.\d+$" -or $scalars["cpython_version"] -notmatch "^\d+\.\d+\.\d+$") {
        Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock contains an invalid version."
    }
    $required = @(
        "windows|x86_64", "windows|aarch64", "macos|x86_64",
        "macos|aarch64", "linux-gnu|x86_64", "linux-gnu|aarch64"
    )
    if ($assets.Keys.Count -ne $required.Count -or @($required | Where-Object { -not $assets.ContainsKey($_) }).Count -ne 0) {
        Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock must contain exactly the six required platform records."
    }
    [pscustomobject]@{
        UvVersion = $scalars["uv_version"]
        CpythonVersion = $scalars["cpython_version"]
        Asset = $assets["$Platform|$Architecture"]
    }
}

function Get-NativeArchitecture {
    foreach ($value in @($env:PROCESSOR_ARCHITEW6432, $env:PROCESSOR_ARCHITECTURE)) {
        switch -Regex ($value) {
            "^(AMD64|x86_64)$" { return "x86_64" }
            "^(ARM64|aarch64)$" { return "aarch64" }
        }
    }
    Stop-Bootstrap "FD-ARCH" "Unsupported Windows architecture."
}

function Test-UvVersion {
    param([string]$UvBinary, [string]$ExpectedVersion)
    if (-not (Test-Path -LiteralPath $UvBinary -PathType Leaf)) { return $false }
    $outputLines = @(& $UvBinary --version 2>$null)
    if ($LASTEXITCODE -ne 0 -or $outputLines.Count -ne 1) { return $false }
    $output = ([string]$outputLines[0]).Trim()
    if ([string]::IsNullOrWhiteSpace($output)) { return $false }
    $pattern = '^uv (?<Version>[0-9]+\.[0-9]+\.[0-9]+)(?: \([^\r\n()]+\))?$'
    $match = [regex]::Match($output, $pattern)
    return ($match.Success -and $match.Groups["Version"].Value -eq $ExpectedVersion)
}

function Test-ManagedPython {
    param([string]$Python, [string]$PythonRoot, [switch]$VirtualEnvironment)
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { return $false }
    $check = if ($VirtualEnvironment) {
        "from pathlib import Path; import sys; root=Path(sys.argv[1]).resolve(); raise SystemExit(0 if sys.version_info[:3] == (3, 12, 13) and sys.implementation.name == 'cpython' and Path(sys.base_prefix).resolve().is_relative_to(root) else 1)"
    }
    else {
        "from pathlib import Path; import sys; root=Path(sys.argv[1]).resolve(); paths=(Path(sys.base_prefix).resolve(), Path(sys.executable).resolve()); raise SystemExit(0 if sys.version_info[:3] == (3, 12, 13) and sys.implementation.name == 'cpython' and all(path.is_relative_to(root) for path in paths) else 1)"
    }
    & $Python -c $check $PythonRoot *> $null
    return $LASTEXITCODE -eq 0
}

$exitCode = 1
$uvEnvironment = @{}
$previousUvEnvironment = $null
try {
    $repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $lockPath = Join-Path $repositoryRoot "environment\bootstrap_toolchain.lock"
    $bootstrapPath = Join-Path $repositoryRoot "scripts\bootstrap_local_environment.py"
    if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
        Stop-Bootstrap "FD-LOCK-MISSING" "Toolchain lock is missing."
    }
    $platform = "windows"
    $architecture = Get-NativeArchitecture
    $lock = Get-Lock -Path $lockPath -Platform $platform -Architecture $architecture
    if ($null -eq $lock.Asset) {
        Stop-Bootstrap "FD-LOCK-MALFORMED" "Toolchain lock has no asset for this platform."
    }

    $toolsRoot = Join-Path $repositoryRoot ".tools"
    $uvVersionRoot = Join-Path $toolsRoot (Join-Path "uv" $lock.UvVersion)
    $platformDirectory = Join-Path $uvVersionRoot "$platform-$architecture"
    $archivePath = Join-Path $platformDirectory $lock.Asset.Filename
    $uvBinary = Join-Path $platformDirectory "uv.exe"
    $pythonRoot = Join-Path $toolsRoot "python"
    $cacheRoot = Join-Path $toolsRoot "cache"
    New-Item -ItemType Directory -Force -Path $uvVersionRoot, $pythonRoot, $cacheRoot | Out-Null

    $archiveMatches = {
        param([string]$Archive)
        if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) { return $false }
        return ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash -eq $lock.Asset.Sha256)
    }
    $validLocalUv = $false
    if (& $archiveMatches $archivePath) {
        if (Test-Path -LiteralPath $uvBinary -PathType Leaf) {
            $validLocalUv = Test-UvVersion -UvBinary $uvBinary -ExpectedVersion $lock.UvVersion
        }
    }
    if (-not $validLocalUv) {
        $stagingDirectory = Join-Path $uvVersionRoot (".staging-" + [guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Path $stagingDirectory | Out-Null
        try {
            $stagingArchive = Join-Path $stagingDirectory $lock.Asset.Filename
            if (& $archiveMatches $archivePath) {
                Copy-Item -LiteralPath $archivePath -Destination $stagingArchive
            }
            else {
                Invoke-WebRequest -Uri "https://github.com/astral-sh/uv/releases/download/$($lock.UvVersion)/$($lock.Asset.Filename)" -OutFile $stagingArchive
            }
            if (-not (& $archiveMatches $stagingArchive)) {
                Stop-Bootstrap "FD-CHECKSUM" "uv archive SHA-256 did not match the toolchain lock."
            }
            $extractDirectory = Join-Path $stagingDirectory "extract"
            $payloadDirectory = Join-Path $stagingDirectory "payload"
            Expand-Archive -LiteralPath $stagingArchive -DestinationPath $extractDirectory -Force
            New-Item -ItemType Directory -Path $payloadDirectory | Out-Null
            $stagedUvCandidates = @(
                Get-ChildItem -LiteralPath $extractDirectory -Recurse -File -Filter "uv.exe"
            )
            if ($stagedUvCandidates.Count -ne 1) {
                Stop-Bootstrap "FD-UV-EXECUTABLE" "Verified uv archive must contain exactly one uv.exe."
            }
            $stagedUv = $stagedUvCandidates[0].FullName
            if (-not (Test-UvVersion -UvBinary $stagedUv -ExpectedVersion $lock.UvVersion)) {
                Stop-Bootstrap "FD-UV-VERSION" "Verified uv archive reported an unexpected version."
            }
            Move-Item -LiteralPath $stagingArchive -Destination (Join-Path $payloadDirectory $lock.Asset.Filename)
            Move-Item -LiteralPath $stagedUv -Destination (Join-Path $payloadDirectory "uv.exe")
            $backupDirectory = $null
            if (Test-Path -LiteralPath $platformDirectory) {
                $backupDirectory = Join-Path $uvVersionRoot (".previous-" + [guid]::NewGuid().ToString("N"))
                Move-Item -LiteralPath $platformDirectory -Destination $backupDirectory
            }
            Move-Item -LiteralPath $payloadDirectory -Destination $platformDirectory
            if ($null -ne $backupDirectory) {
                Remove-Item -LiteralPath $backupDirectory -Recurse -Force
            }
        }
        catch {
            if ($_.Exception.Message -match "^FD-") { throw }
            Stop-Bootstrap "FD-UV-ARCHIVE" "uv download, extraction, or repository-local installation failed; check network, TLS, proxy, disk space, and write access."
        }
        finally {
            if (Test-Path -LiteralPath $stagingDirectory) {
                Remove-Item -LiteralPath $stagingDirectory -Recurse -Force
            }
        }
    }

    if (-not (Test-UvVersion -UvBinary $uvBinary -ExpectedVersion $lock.UvVersion)) {
        Stop-Bootstrap "FD-UV-VERSION" "Repository-local uv reported an unexpected version."
    }

    $uvEnvironment = @{
        UV_PYTHON_INSTALL_DIR = $pythonRoot
        UV_CACHE_DIR = $cacheRoot
        UV_MANAGED_PYTHON = "1"
        UV_PYTHON_DOWNLOADS = "manual"
        UV_PYTHON_INSTALL_BIN = "0"
        UV_PYTHON_INSTALL_REGISTRY = "0"
        UV_NO_CONFIG = "1"
    }
    $previousUvEnvironment = @{}
    foreach ($name in $uvEnvironment.Keys) {
        $previousUvEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
        [Environment]::SetEnvironmentVariable($name, $uvEnvironment[$name], "Process")
    }

    & $uvBinary --no-config python install --no-bin --no-registry "cpython@$($lock.CpythonVersion)"
    if ($LASTEXITCODE -ne 0) { Stop-Bootstrap "FD-PYTHON-INSTALL" "Managed CPython installation failed; check network, TLS, proxy, disk space, and write access." }
    $managedPython = (& $uvBinary --no-config python find --managed-python --system "cpython@$($lock.CpythonVersion)").Trim()
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $managedPython -PathType Leaf)) {
        Stop-Bootstrap "FD-PYTHON-FIND" "Exact managed CPython could not be found."
    }
    if (-not (Test-ManagedPython -Python $managedPython -PythonRoot $pythonRoot)) {
        Stop-Bootstrap "FD-PYTHON-VERIFY" "Managed CPython is not exactly CPython 3.12.13 below .tools/python."
    }

    $venvDirectory = Join-Path $repositoryRoot ".venv"
    $venvPython = Join-Path $venvDirectory "Scripts\python.exe"
    if ((Test-Path -LiteralPath $venvDirectory) -and -not (Test-ManagedPython -Python $venvPython -PythonRoot $pythonRoot -VirtualEnvironment)) {
        if ($venvDirectory -ne (Join-Path $repositoryRoot ".venv")) {
            Stop-Bootstrap "FD-VENV-SAFETY" "Refusing to replace a virtual environment outside the repository root."
        }
        Remove-Item -LiteralPath $venvDirectory -Recurse -Force
    }
    & $managedPython $bootstrapPath
    $exitCode = $LASTEXITCODE
}
catch {
    [Console]::Error.WriteLine("ERROR $($_.Exception.Message)")
    $exitCode = 1
}
finally {
    if ($null -ne $previousUvEnvironment) {
        foreach ($name in $previousUvEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $previousUvEnvironment[$name], "Process")
        }
    }
}
exit $exitCode
