[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteSsh,
    [Parameter(Mandatory = $true)]
    [string]$RemoteRepo,
    [string]$PythonExe = "C:\miniconda3\envs\g1_deploy\python.exe",
    [ValidateRange(1, 10)][int]$SshAttempts = 3,
    [ValidateRange(3, 120)][int]$SshConnectTimeoutSeconds = 15,
    [ValidateRange(1, 1000000)][int]$ScpLimitKbps = 8192,
    [switch]$CheckOnly,
    [switch]$SkipRemoteExport,
    [switch]$AdoptExisting,
    [string]$AdoptMetadataPath
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

if ($CheckOnly -and $AdoptExisting) {
    throw "-CheckOnly and -AdoptExisting are mutually exclusive."
}
if ($AdoptMetadataPath -and -not $AdoptExisting) {
    throw "-AdoptMetadataPath is only valid together with -AdoptExisting."
}
if ($RemoteRepo -notmatch '^/[A-Za-z0-9._/-]+$') {
    throw "Unsafe -RemoteRepo value: $RemoteRepo"
}
if ($RemoteSsh -notmatch '^[A-Za-z0-9._-]+@[A-Za-z0-9._:-]+$') {
    throw "Unsafe -RemoteSsh value: $RemoteSsh"
}

$script:RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$script:DailyRoot = [IO.Path]::GetFullPath((Join-Path $script:RepoRoot "models\kengo_sonic_daily"))
$script:DailyPrefix = $script:DailyRoot.TrimEnd('\') + '\'
$script:StateDir = Join-Path $script:DailyRoot "state"
$script:RunsDir = Join-Path $script:DailyRoot "runs"
$script:StagingRoot = Join-Path $script:DailyRoot "staging"
$script:ReferenceCache = Join-Path $script:DailyRoot "cache\references"
$script:LogsDir = Join-Path $script:DailyRoot "logs"
$script:ActiveStaging = $null
$script:ActivePromotedRun = $null
$script:LogPath = $null
$script:Mutex = $null
$script:MutexAcquired = $false
$script:OwnershipMarker = '.kengo_daily_owned'

function Test-ManagedPath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    return $full.Equals($script:DailyRoot, [StringComparison]::OrdinalIgnoreCase) -or
        $full.StartsWith($script:DailyPrefix, [StringComparison]::OrdinalIgnoreCase)
}

function Remove-ManagedDirectory([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not (Test-ManagedPath $full) -or $full.Equals($script:DailyRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to recursively delete unmanaged path: $full"
    }
    $marker = Join-Path $full $script:OwnershipMarker
    if (-not [IO.File]::Exists($marker)) {
        throw "Refusing to recursively delete an unmarked directory: $full"
    }
    if ([IO.Directory]::Exists($full)) {
        $cursor = New-Object IO.DirectoryInfo($full)
        while ($null -ne $cursor) {
            if (($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Refusing to delete through a reparse point: $($cursor.FullName)"
            }
            if ($cursor.FullName.Equals($script:DailyRoot, [StringComparison]::OrdinalIgnoreCase)) { break }
            $cursor = $cursor.Parent
        }
        if ($null -eq $cursor) { throw "Managed delete path has no daily-root ancestor: $full" }
        $pending = New-Object 'Collections.Generic.Stack[string]'
        $pending.Push($full)
        while ($pending.Count -gt 0) {
            $directory = $pending.Pop()
            foreach ($entry in [IO.Directory]::EnumerateFileSystemEntries($directory)) {
                $attributes = [IO.File]::GetAttributes($entry)
                if (($attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                    throw "Refusing to delete a tree containing a reparse point: $entry"
                }
                if (($attributes -band [IO.FileAttributes]::Directory) -ne 0) { $pending.Push($entry) }
            }
        }
        [IO.Directory]::Delete($full, $true)
    }
}

function Test-OwnedDirectory([string]$Path) {
    if (-not (Test-ManagedPath $Path)) { return $false }
    return [IO.File]::Exists((Join-Path ([IO.Path]::GetFullPath($Path)) $script:OwnershipMarker))
}

function Add-OwnershipMarker([string]$Directory) {
    if (-not (Test-ManagedPath $Directory)) { throw "Refusing to mark unmanaged directory: $Directory" }
    [IO.Directory]::CreateDirectory($Directory) | Out-Null
    $marker = Join-Path $Directory $script:OwnershipMarker
    [IO.File]::WriteAllText($marker, "owned by run_kengo_daily_sim2sim.ps1`r`n", (New-Object Text.UTF8Encoding($false)))
}

function Write-Log([string]$Level, [string]$Message) {
    $line = "{0} [{1}] {2}" -f ([DateTime]::UtcNow.ToString("o")), $Level.ToUpperInvariant(), $Message
    Write-Host $line
    if ($script:LogPath) {
        [IO.File]::AppendAllText($script:LogPath, $line + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
    }
}

function Write-JsonAtomic($Value, [string]$Path) {
    if (-not (Test-ManagedPath $Path)) { throw "Refusing to write state outside daily root: $Path" }
    $parent = Split-Path -Parent $Path
    [IO.Directory]::CreateDirectory($parent) | Out-Null
    $temporary = Join-Path $parent (".{0}.{1}.tmp" -f ([IO.Path]::GetFileName($Path)), [Guid]::NewGuid().ToString("N"))
    $json = $Value | ConvertTo-Json -Depth 20
    [IO.File]::WriteAllText($temporary, $json + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
    try {
        if ([IO.File]::Exists($Path)) {
            [IO.File]::Replace($temporary, $Path, $null, $true)
        } else {
            [IO.File]::Move($temporary, $Path)
        }
    } finally {
        if ([IO.File]::Exists($temporary)) { [IO.File]::Delete($temporary) }
    }
}

function Read-Json([string]$Path) {
    if (-not [IO.File]::Exists($Path)) { return $null }
    return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-HexSha([string]$Value, [string]$Name) {
    if ($Value -notmatch '^[0-9a-fA-F]{64}$') { throw "$Name is not a SHA-256 digest: $Value" }
}

function Invoke-SshRetry([string]$RemoteCommand, [string]$Purpose) {
    $ssh = (Get-Command ssh.exe -ErrorAction Stop).Source
    $common = @(
        '-o', 'BatchMode=yes',
        '-o', 'Compression=no',
        '-o', "ConnectTimeout=$SshConnectTimeoutSeconds",
        '-o', 'ConnectionAttempts=1',
        '-o', 'ServerAliveInterval=15',
        '-o', 'ServerAliveCountMax=2',
        '-o', 'StrictHostKeyChecking=yes',
        '-o', 'LogLevel=ERROR'
    )
    for ($attempt = 1; $attempt -le $SshAttempts; $attempt++) {
        Write-Log INFO "$Purpose (SSH attempt $attempt/$SshAttempts)"
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $output = @(& $ssh @common $RemoteSsh $RemoteCommand 2>&1)
            $code = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $savedPreference
        }
        foreach ($line in $output) { Write-Log REMOTE ([string]$line) }
        if ($code -eq 0) { return $output }
        if ($attempt -lt $SshAttempts) { Start-Sleep -Seconds ([Math]::Min(5 * $attempt, 15)) }
    }
    throw "$Purpose failed after $SshAttempts SSH attempt(s)."
}

function Get-RemoteBest {
    $command = "cd '$RemoteRepo' && printf '__KENGO_BEST_BEGIN__\n' && cat checkpoint_backups/hourly_best/best_so_far.json && printf '\n__KENGO_BEST_END__\n'"
    $text = (Invoke-SshRetry $command "Read remote best metadata") -join "`n"
    $match = [regex]::Match($text, '(?s)__KENGO_BEST_BEGIN__\s*(?<json>\{.*?\})\s*__KENGO_BEST_END__')
    if (-not $match.Success) { throw "Remote best metadata markers were not found." }
    $best = $match.Groups['json'].Value | ConvertFrom-Json
    $step = [int64]$best.step
    $sha = ([string]$best.sha256).ToLowerInvariant()
    if ($step -lt 1) { throw "Remote best step is invalid: $step" }
    Assert-HexSha $sha "Remote checkpoint SHA"
    return [pscustomobject]@{
        step = $step
        checkpoint_sha256 = $sha
        mean_reward = $best.mean_reward
        selected_at = $best.selected_at
        bytes = $best.bytes
    }
}

function Read-AdoptMetadata([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not [IO.File]::Exists($full)) { throw "Adoption metadata is missing: $full" }
    $metadata = Get-Content -LiteralPath $full -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($required in @('schema_version','step','checkpoint_sha256','onnx_sha256','bytes','remote_onnx_path')) {
        if ($null -eq $metadata.PSObject.Properties[$required]) { throw "Adoption metadata is missing required field '$required'." }
    }
    if (($metadata.schema_version -isnot [int]) -and ($metadata.schema_version -isnot [long])) { throw "Adoption metadata schema_version must be an integer." }
    if ([int]$metadata.schema_version -ne 1) { throw "Unsupported adoption metadata schema_version: $($metadata.schema_version)" }
    if (($metadata.step -isnot [int]) -and ($metadata.step -isnot [long])) { throw "Adoption metadata step must be an integer." }
    $step = [int64]$metadata.step
    if ($step -lt 1) { throw "Adoption metadata step is invalid." }
    $checkpointSha = ([string]$metadata.checkpoint_sha256).ToLowerInvariant()
    $onnxSha = ([string]$metadata.onnx_sha256).ToLowerInvariant()
    Assert-HexSha $checkpointSha "Adoption checkpoint SHA"
    Assert-HexSha $onnxSha "Adoption ONNX SHA"
    if (($metadata.bytes -isnot [int]) -and ($metadata.bytes -isnot [long])) { throw "Adoption metadata bytes must be an integer." }
    $bytes = [int64]$metadata.bytes
    if ($bytes -lt 1) { throw "Adoption metadata byte count is invalid." }
    $stepText = '{0:D6}' -f $step
    $canonical = "$RemoteRepo/sim2sim_exports/kengo_best_step_$step/model_step_$($stepText)_kengo.onnx"
    if ([string]$metadata.remote_onnx_path -cne $canonical) { throw "Adoption metadata remote_onnx_path is not canonical: $($metadata.remote_onnx_path)" }
    return [pscustomobject]@{
        best = [pscustomobject]@{ step=$step; checkpoint_sha256=$checkpointSha; mean_reward=$null; selected_at=$null; bytes=$null }
        onnx = [pscustomobject]@{ step=$step; path=$canonical; sha256=$onnxSha; bytes=$bytes }
        source_path = $full
    }
}

function Test-SameBest($Left, $Right) {
    if ($null -eq $Left -or $null -eq $Right) { return $false }
    return ([int64]$Left.step -eq [int64]$Right.step) -and
        (([string]$Left.checkpoint_sha256).ToLowerInvariant() -eq ([string]$Right.checkpoint_sha256).ToLowerInvariant())
}

function Get-RemoteOnnxInfo($Best) {
    $stepText = '{0:D6}' -f [int64]$Best.step
    $path = "$RemoteRepo/sim2sim_exports/kengo_best_step_$([int64]$Best.step)/model_step_$($stepText)_kengo.onnx"
    $command = "set -eu; p='$path'; test -s `"`$p`"; printf '__KENGO_ONNX_BEGIN__\n'; sha256sum -- `"`$p`"; stat -c '%s' -- `"`$p`"; printf '__KENGO_ONNX_END__\n'"
    $text = (Invoke-SshRetry $command "Inspect remote ONNX") -join "`n"
    $match = [regex]::Match($text, '(?s)__KENGO_ONNX_BEGIN__\s*(?<sha>[0-9a-fA-F]{64})\s+[^\r\n]+[\r\n]+(?<bytes>[0-9]+)\s*__KENGO_ONNX_END__')
    if (-not $match.Success) { throw "Remote ONNX metadata markers were not found." }
    $sha = $match.Groups['sha'].Value.ToLowerInvariant()
    Assert-HexSha $sha "Remote ONNX SHA"
    $bytes = [int64]$match.Groups['bytes'].Value
    if ($bytes -lt 1) { throw "Remote ONNX is empty: $path" }
    return [pscustomobject]@{ step = [int64]$Best.step; path = $path; sha256 = $sha; bytes = $bytes }
}

function Resolve-RemoteOnnx($InitialBest) {
    $best = $InitialBest
    if ($SkipRemoteExport) { return [pscustomobject]@{ best = $best; onnx = (Get-RemoteOnnxInfo $best) } }
    for ($round = 1; $round -le 2; $round++) {
        Ensure-RemoteHelper
        $helper = "cd '$RemoteRepo' && timeout --signal=TERM --kill-after=30s 20m bash gear_sonic/scripts/export_latest_kengo_best_onnx.sh"
        Invoke-SshRetry $helper "Export latest best Kengo ONNX" | Out-Null
        $after = Get-RemoteBest
        if (-not (Test-SameBest $best $after)) {
            Write-Log WARN "Remote best changed while exporting; retrying against step $($after.step)."
            $best = $after
            continue
        }
        return [pscustomobject]@{ best = $after; onnx = (Get-RemoteOnnxInfo $after) }
    }
    throw "Remote best changed during both export attempts."
}

function Get-RemoteHelperInfo {
    $path = "$RemoteRepo/gear_sonic/scripts/export_latest_kengo_best_onnx.sh"
    $command = "set -eu; p='$path'; if [ -L `"`$p`" ]; then printf '__KENGO_HELPER_INVALID__\n'; elif [ -f `"`$p`" ]; then printf '__KENGO_HELPER_BEGIN__\n'; sha256sum -- `"`$p`"; stat -c '%s' -- `"`$p`"; printf '__KENGO_HELPER_END__\n'; elif [ -e `"`$p`" ]; then printf '__KENGO_HELPER_INVALID__\n'; else printf '__KENGO_HELPER_MISSING__\n'; fi"
    $text = (Invoke-SshRetry $command "Inspect remote export helper") -join "`n"
    if ($text -match '__KENGO_HELPER_INVALID__') { throw "Remote helper path exists but is not a regular non-symlink file: $path" }
    if ($text -match '__KENGO_HELPER_MISSING__') { return [pscustomobject]@{ exists=$false; path=$path; sha256=$null; bytes=$null } }
    $match = [regex]::Match($text, '(?s)__KENGO_HELPER_BEGIN__\s*(?<sha>[0-9a-fA-F]{64})\s+[^\r\n]+[\r\n]+(?<bytes>[0-9]+)\s*__KENGO_HELPER_END__')
    if (-not $match.Success) { throw "Could not parse remote helper metadata." }
    return [pscustomobject]@{ exists=$true; path=$path; sha256=$match.Groups['sha'].Value.ToLowerInvariant(); bytes=[int64]$match.Groups['bytes'].Value }
}

function Ensure-RemoteHelper {
    $local = Join-Path $script:RepoRoot 'gear_sonic\scripts\export_latest_kengo_best_onnx.sh'
    if (-not [IO.File]::Exists($local)) { throw "Local remote-export helper is missing: $local" }
    $localSha = Get-Sha256 $local
    $localBytes = (Get-Item -LiteralPath $local).Length
    $remote = Get-RemoteHelperInfo
    if ($remote.exists) {
        if ($remote.sha256 -ne $localSha -or [int64]$remote.bytes -ne [int64]$localBytes) {
            throw "Remote export helper differs from the local helper; refusing to overwrite it."
        }
        Write-Log INFO "Remote export helper already matches local SHA $localSha."
        return
    }
    $token = [Guid]::NewGuid().ToString('N')
    $temporary = "$RemoteRepo/gear_sonic/scripts/.export_latest_kengo_best_onnx.$($localSha.Substring(0,12)).$token.tmp"
    $scp = (Get-Command scp.exe -ErrorAction Stop).Source
    $arguments = @('-B','-q','-l',[string]$ScpLimitKbps,'-o','BatchMode=yes','-o','Compression=no','-o',"ConnectTimeout=$SshConnectTimeoutSeconds",'-o','ConnectionAttempts=1','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','StrictHostKeyChecking=yes',$local,("{0}:{1}" -f $RemoteSsh,$temporary))
    Invoke-NativeChecked $scp $arguments "Deploy one temporary remote-export helper file" | Out-Null
    $publish = "set -eu; tmp='$temporary'; dst='$($remote.path)'; lock='$RemoteRepo/gear_sonic/scripts/.export_latest_kengo_best_onnx.deploy.lock'; trap 'rm -f -- `"`$tmp`"' EXIT; test -f `"`$tmp`"; test ! -L `"`$tmp`"; actual=`$(sha256sum -- `"`$tmp`" | cut -d' ' -f1); bytes=`$(stat -c '%s' -- `"`$tmp`"); test `"`$actual`" = '$localSha'; test `"`$bytes`" = '$localBytes'; exec 8>`"`$lock`"; flock -x 8; if [ -e `"`$dst`" ] || [ -L `"`$dst`" ]; then test -f `"`$dst`"; test ! -L `"`$dst`"; existing=`$(sha256sum -- `"`$dst`" | cut -d' ' -f1); test `"`$existing`" = '$localSha'; exit 0; fi; chmod 700 `"`$tmp`"; mv -- `"`$tmp`" `"`$dst`"; trap - EXIT"
    Invoke-SshRetry $publish "Verify and atomically publish remote export helper" | Out-Null
    $after = Get-RemoteHelperInfo
    if (-not $after.exists -or $after.sha256 -ne $localSha -or [int64]$after.bytes -ne [int64]$localBytes) { throw "Remote helper post-deployment verification failed." }
}

function Invoke-NativeChecked([string]$Executable, [object[]]$Arguments, [string]$Purpose, [int[]]$AllowedExitCodes = @(0)) {
    Write-Log INFO $Purpose
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = @(& $Executable @Arguments 2>&1)
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
    }
    foreach ($line in $output) { Write-Log PROCESS ([string]$line) }
    if ($AllowedExitCodes -notcontains $code) { throw "$Purpose failed with exit code $code." }
    return [pscustomobject]@{ ExitCode = $code; Output = $output }
}

function Invoke-PythonJson([string]$Code, [object[]]$Arguments, [string]$Purpose) {
    $result = Invoke-NativeChecked $PythonExe (@('-c', $Code) + $Arguments) $Purpose
    $jsonLine = @($result.Output | ForEach-Object { [string]$_ } | Where-Object { $_.TrimStart().StartsWith('{') }) | Select-Object -Last 1
    if (-not $jsonLine) { throw "$Purpose did not emit JSON." }
    return ($jsonLine | ConvertFrom-Json)
}

$script:OnnxValidator = @'
import json, sys
import numpy as np
import onnx
import onnxruntime as ort
p, seed = sys.argv[1], int(sys.argv[2])
onnx.checker.check_model(onnx.load(p))
s = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
i, o = s.get_inputs()[0], s.get_outputs()[0]
assert i.name == 'obs_dict' and list(i.shape) == [1, 1270] and i.type == 'tensor(float)'
assert o.name == 'action' and list(o.shape) == [1, 23] and o.type == 'tensor(float)'
rng = np.random.default_rng(seed)
for x in (np.zeros((1, 1270), np.float32), rng.standard_normal((1, 1270)).astype(np.float32)):
    y = s.run([o.name], {i.name: x})[0]
    assert y.shape == (1, 23) and np.isfinite(y).all()
print(json.dumps({'checker':'ok','provider':s.get_providers()[0],'input':[i.name,list(i.shape),i.type],'output':[o.name,list(o.shape),o.type]}))
'@

$script:VideoValidator = @'
import cv2, json, os, sys
p, expected, fps, width, height = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
c = cv2.VideoCapture(p)
reported, got_fps = int(round(c.get(cv2.CAP_PROP_FRAME_COUNT))), float(c.get(cv2.CAP_PROP_FPS))
n = 0
shape_ok = True
while True:
    ok, frame = c.read()
    if not ok: break
    shape_ok = shape_ok and frame.shape[1] == width and frame.shape[0] == height and frame.size > 0
    n += 1
c.release()
valid = os.path.getsize(p) > 0 and reported == n == expected and abs(got_fps-fps) < 1e-6 and shape_ok
print(json.dumps({'path':p,'reported_frames':reported,'decoded_frames':n,'expected_frames':expected,'fps':got_fps,'width':width,'height':height,'valid':valid}))
raise SystemExit(0 if valid else 2)
'@

function Test-Video([string]$Path, [int]$Frames, [int]$Width, [int]$Height, [string]$Purpose) {
    if (-not [IO.File]::Exists($Path)) { throw "$Purpose is missing: $Path" }
    return Invoke-PythonJson $script:VideoValidator @($Path, $Frames, 30, $Width, $Height) $Purpose
}

function Test-FullReferenceResult([string]$MetricsPath, [string]$VideoPath, [string]$Purpose) {
    $metrics = Read-Json $MetricsPath
    if ($null -eq $metrics) { throw "$Purpose metrics are missing: $MetricsPath" }
    if ($null -ne $metrics.runtime.runtime_error -and ([string]$metrics.runtime.runtime_error).Length -gt 0) { throw "$Purpose has runtime_error: $($metrics.runtime.runtime_error)" }
    if (-not [bool]$metrics.stability.finite) { throw "$Purpose produced non-finite state." }
    $motionFrames = [int]$metrics.motion.frames
    if ($motionFrames -lt 1 -or [int]$metrics.motion.start_frame -ne 0 -or [int]$metrics.motion.final_frame -ne ($motionFrames - 1) -or [bool]$metrics.motion.loop) { throw "$Purpose did not cover the complete non-looping reference." }
    if ([int]$metrics.runtime.policy_steps -ne $motionFrames -or [int]$metrics.runtime.physics_steps -ne (4 * $motionFrames)) { throw "$Purpose has an incomplete step budget." }
    $allFailures = @($metrics.failures | Where-Object { $_ })
    if ([bool]$metrics.success -ne ($allFailures.Count -eq 0)) { throw "$Purpose success flag disagrees with its failures list." }
    $unexpected = @($allFailures | Where-Object { ([string]$_) -notmatch '^base height fell below ' })
    if ($unexpected.Count -gt 0) { throw "$Purpose has unexpected failure(s): $($unexpected -join '; ')" }
    $videoFrames = [int]$metrics.video.frames
    $expectedVideoFrames = [int][Math]::Ceiling((3.0 * $motionFrames) / 5.0)
    if ($videoFrames -ne $expectedVideoFrames -or [double]$metrics.video.fps -ne 30.0) { throw "$Purpose has invalid or truncated video metadata (expected $expectedVideoFrames frames, got $videoFrames)." }
    Test-Video $VideoPath $videoFrames 1280 720 "$Purpose video decode" | Out-Null
    return [pscustomobject]@{ metrics = $metrics; video_frames = $videoFrames; motion_frames = $motionFrames }
}

function Get-FreeBytes {
    $root = [IO.Path]::GetPathRoot($script:DailyRoot)
    return (New-Object IO.DriveInfo($root)).AvailableFreeSpace
}

function Get-Ffmpeg {
    $code = 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'
    $result = Invoke-NativeChecked $PythonExe @('-c', $code) "Locate FFmpeg"
    $path = @($result.Output | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ }) | Select-Object -Last 1
    if (-not $path -or -not [IO.File]::Exists($path)) { throw "imageio-ffmpeg executable was not found." }
    return $path
}

function Get-UniqueFile([string]$Directory, [string]$Filter, [string]$Purpose) {
    $matches = @(Get-ChildItem -LiteralPath $Directory -Filter $Filter -File -ErrorAction SilentlyContinue)
    if ($matches.Count -ne 1) { throw "$Purpose expected one '$Filter' file, found $($matches.Count)." }
    return $matches[0].FullName
}

function Get-Artifact($Manifest, [string]$Key) {
    $property = $Manifest.artifacts.PSObject.Properties[$Key]
    if ($null -eq $property) { throw "Previous manifest has no '$Key' artifact." }
    return $property.Value
}

function Ensure-ReferenceVideo($Spec, [int]$ExpectedFrames, $FallbackArtifact) {
    $motionSha = Get-Sha256 $Spec.Motion
    if ($null -ne $FallbackArtifact -and ([string]$FallbackArtifact.motion_sha256).ToLowerInvariant() -eq $motionSha) {
        $fallback = [string]$FallbackArtifact.reference_video
        if ([IO.File]::Exists($fallback)) {
            Test-Video $fallback $ExpectedFrames 1280 720 "$($Spec.Key) reused reference" | Out-Null
            return $fallback
        }
    }
    $video = Join-Path $script:ReferenceCache "$($Spec.Key).reference.mp4"
    $index = Join-Path $script:ReferenceCache "$($Spec.Key).reference.json"
    $cached = Read-Json $index
    $xmlSha = Get-Sha256 $script:XmlPath
    $rendererSha = Get-Sha256 $script:ReferenceRenderer
    if ($null -ne $cached -and ([string]$cached.motion_sha256).ToLowerInvariant() -eq $motionSha -and ([string]$cached.xml_sha256).ToLowerInvariant() -eq $xmlSha -and ([string]$cached.renderer_sha256).ToLowerInvariant() -eq $rendererSha -and [int]$cached.video_frames -eq $ExpectedFrames -and [IO.File]::Exists($video)) {
        Test-Video $video $ExpectedFrames 1280 720 "$($Spec.Key) cached reference" | Out-Null
        return $video
    }
    $temporary = "$video.partial.mp4"
    if ([IO.File]::Exists($temporary)) { [IO.File]::Delete($temporary) }
    Invoke-NativeChecked $PythonExe @($script:ReferenceRenderer, '--motion', $Spec.Motion, '--xml', $script:XmlPath, '--video', $temporary) "Render $($Spec.Key) full reference" | Out-Null
    Test-Video $temporary $ExpectedFrames 1280 720 "$($Spec.Key) rendered reference" | Out-Null
    if ([IO.File]::Exists($video)) { [IO.File]::Replace($temporary, $video, $null, $true) } else { [IO.File]::Move($temporary, $video) }
    Write-JsonAtomic ([ordered]@{ schema_version=1; motion_sha256=$motionSha; xml_sha256=$xmlSha; renderer_sha256=$rendererSha; video_frames=$ExpectedFrames; video_path=$video }) $index
    return $video
}

function New-Comparison([string]$Ffmpeg, [string]$Reference, [string]$Previous, [string]$Current, [int]$Frames, [int64]$PreviousStep, [int64]$CurrentStep, [string]$Output) {
    Test-Video $Reference $Frames 1280 720 "Reference comparison input" | Out-Null
    Test-Video $Previous $Frames 1280 720 "Previous comparison input" | Out-Null
    Test-Video $Current $Frames 1280 720 "Current comparison input" | Out-Null
    $temporary = "$Output.partial.mp4"
    if ([IO.File]::Exists($temporary)) { [IO.File]::Delete($temporary) }
    $filter = "[0:v]trim=start_frame=0:end_frame=$Frames,setpts=N/(30*TB),scale=960:540:flags=lanczos,pad=960:600:0:60:color=black,drawtext=text='REFERENCE | FULL REF':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=15[ref];[1:v]trim=start_frame=0:end_frame=$Frames,setpts=N/(30*TB),scale=960:540:flags=lanczos,pad=960:600:0:60:color=black,drawtext=text='STEP $PreviousStep | PREVIOUS | FULL REF':fontcolor=yellow:fontsize=28:x=(w-text_w)/2:y=15[old];[2:v]trim=start_frame=0:end_frame=$Frames,setpts=N/(30*TB),scale=960:540:flags=lanczos,pad=960:600:0:60:color=black,drawtext=text='STEP $CurrentStep | NEW | FULL REF':fontcolor=lime:fontsize=28:x=(w-text_w)/2:y=15[new];[ref][old][new]hstack=inputs=3:shortest=1[v]"
    Invoke-NativeChecked $Ffmpeg @('-nostdin','-hide_banner','-loglevel','warning','-i',$Reference,'-i',$Previous,'-i',$Current,'-filter_complex',$filter,'-map','[v]','-an','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-fps_mode','cfr','-r','30','-movflags','+faststart','-frames:v',[string]$Frames,'-y',$temporary) "Encode three-panel comparison" | Out-Null
    Test-Video $temporary $Frames 2880 600 "Three-panel comparison decode" | Out-Null
    [IO.File]::Move($temporary, $Output)
}

function Promote-Run($Manifest, [string]$Stage, [string]$Final, $OldCurrent) {
    if ([IO.Directory]::Exists($Final)) { throw "Final run directory already exists: $Final" }
    [IO.Directory]::Move($Stage, $Final)
    $script:ActiveStaging = $null
    $script:ActivePromotedRun = $Final
    if ($null -ne $OldCurrent) { Write-JsonAtomic $OldCurrent (Join-Path $script:StateDir 'previous.json') }
    $state = [ordered]@{ schema_version=1; run_id=$Manifest.run_id; step=$Manifest.step; checkpoint_sha256=$Manifest.checkpoint_sha256; onnx_sha256=$Manifest.onnx_sha256; manifest_path=(Join-Path $Final 'manifest.json'); model_path=$Manifest.model_path; completed_at=$Manifest.completed_at }
    Write-JsonAtomic $state (Join-Path $script:StateDir 'current.json')
    $script:ActivePromotedRun = $null
    return $state
}

function Invoke-Retention {
    foreach ($directory in @(Get-ChildItem -LiteralPath $script:StagingRoot -Directory -Force -ErrorAction SilentlyContinue)) {
        if ($script:ActiveStaging -and $directory.FullName.Equals($script:ActiveStaging, [StringComparison]::OrdinalIgnoreCase)) { continue }
        if (Test-OwnedDirectory $directory.FullName) {
            Write-Log INFO "Removing marked stale staging directory: $($directory.FullName)"
            Remove-ManagedDirectory $directory.FullName
        } else {
            Write-Log WARN "Skipping unmarked staging directory during retention: $($directory.FullName)"
        }
    }
    $successful = @()
    foreach ($directory in @(Get-ChildItem -LiteralPath $script:RunsDir -Directory -ErrorAction SilentlyContinue)) {
        if (-not (Test-OwnedDirectory $directory.FullName)) {
            Write-Log WARN "Skipping unmarked directory during retention: $($directory.FullName)"
            continue
        }
        try {
            $manifest = Read-Json (Join-Path $directory.FullName 'manifest.json')
            if ($null -ne $manifest -and [string]$manifest.status -eq 'complete') {
                $successful += [pscustomobject]@{ Directory=$directory; Completed=[DateTime]::Parse([string]$manifest.completed_at).ToUniversalTime() }
            } else { Remove-ManagedDirectory $directory.FullName }
        } catch { Remove-ManagedDirectory $directory.FullName }
    }
    $keep = @($successful | Sort-Object Completed -Descending | Select-Object -First 2 | ForEach-Object { $_.Directory.FullName })
    foreach ($item in $successful) { if ($keep -notcontains $item.Directory.FullName) { Remove-ManagedDirectory $item.Directory.FullName } }
    $logs = @(Get-ChildItem -LiteralPath $script:LogsDir -File -Filter '*.log' | Sort-Object LastWriteTimeUtc -Descending)
    foreach ($old in @($logs | Select-Object -Skip 30)) { if (Test-ManagedPath $old.FullName) { [IO.File]::Delete($old.FullName) } }
}

$exitCode = 0
try {
    $script:Mutex = New-Object Threading.Mutex($false, 'Global\KengoSonicDailySim2Sim_v1')
    try { $script:MutexAcquired = $script:Mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $script:MutexAcquired = $true }
    if (-not $script:MutexAcquired) { throw "Another Kengo daily sim2sim run is active." }

    foreach ($directory in @($script:DailyRoot,$script:StateDir,$script:RunsDir,$script:StagingRoot,$script:ReferenceCache,$script:LogsDir)) { [IO.Directory]::CreateDirectory($directory) | Out-Null }
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $script:LogPath = Join-Path $script:LogsDir ("daily_{0}_{1}.log" -f $stamp, $PID)
    [IO.File]::WriteAllText($script:LogPath, '', (New-Object Text.UTF8Encoding($false)))
    Write-Log INFO "Kengo daily sim2sim started. CheckOnly=$CheckOnly SkipRemoteExport=$SkipRemoteExport AdoptExisting=$AdoptExisting ScpLimitKbps=$ScpLimitKbps"

    $currentPath = Join-Path $script:StateDir 'current.json'
    $current = Read-Json $currentPath
    $offlineAdoption = $null
    if ($AdoptExisting -and $AdoptMetadataPath) {
        $offlineAdoption = Read-AdoptMetadata $AdoptMetadataPath
        $remoteBest = $offlineAdoption.best
        Write-Log WARN "Using explicitly supplied offline adoption metadata: $($offlineAdoption.source_path)"
    } else {
        $remoteBest = Get-RemoteBest
    }
    $changed = -not (Test-SameBest $current $remoteBest)
    Write-Log INFO "Remote best step=$($remoteBest.step) checkpoint_sha256=$($remoteBest.checkpoint_sha256) changed=$changed"

    if ($CheckOnly) {
        $check = [ordered]@{ schema_version=1; checked_at=[DateTime]::UtcNow.ToString('o'); remote_step=$remoteBest.step; remote_checkpoint_sha256=$remoteBest.checkpoint_sha256; local_step=if($null -eq $current){$null}else{$current.step}; local_checkpoint_sha256=if($null -eq $current){$null}else{$current.checkpoint_sha256}; changed=$changed }
        Write-Host ("[CHECK_JSON] " + ($check | ConvertTo-Json -Compress))
        Write-Log INFO "Check-only completed; no download, export, simulation, encoding, state update, or cleanup was performed."
    } elseif (-not $changed -and -not $AdoptExisting) {
        if ($null -eq $current -or -not [IO.File]::Exists([string]$current.manifest_path)) { throw "Current state is incomplete even though the remote identity matches." }
        Write-Log INFO "Remote best is already the tested current version; nothing to do."
        Invoke-Retention
    } else {
        if ($null -eq $current -and -not $AdoptExisting) { throw "No current baseline state exists. Run once with -AdoptExisting before normal daily operation." }
        if ($AdoptExisting -and $null -ne $current) { throw "-AdoptExisting is only valid when current.json does not exist." }
        $resolved = if ($null -ne $offlineAdoption) { $offlineAdoption } else { Resolve-RemoteOnnx $remoteBest }
        $remoteBest = $resolved.best
        $remoteOnnx = $resolved.onnx
        if ($AdoptExisting -and -not $changed) { throw "Existing state already represents the remote best; adoption is unnecessary." }

        if (-not [IO.File]::Exists($PythonExe)) { throw "Python interpreter is missing: $PythonExe" }
        $script:Runner = Join-Path $script:RepoRoot 'gear_sonic\scripts\run_kengo_sonic_sim2sim.py'
        $script:ReferenceRenderer = Join-Path $script:RepoRoot 'gear_sonic\scripts\render_kengo_reference_video.py'
        $script:XmlPath = Join-Path $script:RepoRoot 'external_dependencies\kengo_robot_description\xml\kengo_with_fist.xml'
        foreach ($required in @($script:Runner,$script:ReferenceRenderer,$script:XmlPath)) { if (-not [IO.File]::Exists($required)) { throw "Required file is missing: $required" } }
        Invoke-NativeChecked $PythonExe @('-c',"import cv2, imageio_ffmpeg, mujoco, numpy, onnx, onnxruntime; print('dependencies-ok')") "Validate local dependencies" | Out-Null

        $bestRoot = Join-Path $script:RepoRoot 'models\kengo_sonic_best'
        $refsRoot = Join-Path $bestRoot 'references'
        $specs = @(
            [pscustomobject]@{ Key='cmu_56_03'; Motion=(Join-Path $refsRoot 'twist_07265__56_03__fef049ae17_kengo_wbt.npz'); LegacyMetric='cmu_56_03'; LegacyReference='cmu_56_03_fullref_reference.mp4'; LegacyComparison='cmu_56_03_reference_step*_vs_step{0}_fullref.mp4' },
            [pscustomobject]@{ Key='talking_whispering'; Motion=(Join-Path $refsRoot 'twist_01073__Subject3_4_Carine_INF_TalkingWhispering_S3S4_02__a4fde03864_kengo_wbt.npz'); LegacyMetric='talking_whispering'; LegacyReference='talking_whispering_fullref_reference.mp4'; LegacyComparison='talking_whispering_reference_step*_vs_step{0}_fullref.mp4' },
            [pscustomobject]@{ Key='vasso_satisfied'; Motion=(Join-Path $refsRoot 'twist_02387__Vasso_Satisfied_01__8dc0003e31_kengo_wbt.npz'); LegacyMetric='vasso_satisfied'; LegacyReference='vasso_satisfied_fullref_reference.mp4'; LegacyComparison='vasso_reference_step*_vs_step{0}_fullref.mp4' }
        )
        foreach ($spec in $specs) { if (-not [IO.File]::Exists($spec.Motion)) { throw "Reference motion is missing: $($spec.Motion)" } }

        $stepText = '{0:D6}' -f [int64]$remoteBest.step
        $runId = "step$stepText-$($remoteOnnx.sha256.Substring(0,12))-$stamp"
        $stage = Join-Path $script:StagingRoot $runId
        $final = Join-Path $script:RunsDir $runId
        Add-OwnershipMarker $stage
        $script:ActiveStaging = $stage
        $finalModel = if($AdoptExisting){Join-Path $bestRoot "model_step_$($stepText)_kengo.onnx"}else{Join-Path $final "model_step_$($stepText)_kengo.onnx"}
        $artifacts = [ordered]@{}
        $previousStep = $null

        if ($AdoptExisting) {
            if (-not [IO.File]::Exists($finalModel)) { throw "Top-level ONNX for adoption is missing: $finalModel" }
            if ((Get-Item -LiteralPath $finalModel).Length -ne [int64]$remoteOnnx.bytes -or (Get-Sha256 $finalModel) -ne $remoteOnnx.sha256) { throw "Top-level ONNX size or SHA does not match the adopted ONNX metadata." }
            Invoke-PythonJson $script:OnnxValidator @($finalModel, [string]$remoteBest.step) "Validate adopted ONNX" | Out-Null
            $videoRoot = Join-Path $bestRoot 'sim2sim_videos'
            $legacyRefRoot = Join-Path $bestRoot 'sim2sim_references'
            $legacyCompareRoot = Join-Path $bestRoot 'sim2sim_fullref_comparisons'
            $previousSteps = @()
            foreach ($spec in $specs) {
                $metric = Get-UniqueFile $videoRoot ("{0}_step{1}_fullref_*_metrics.json" -f $spec.LegacyMetric,$remoteBest.step) "$($spec.Key) adoption metrics"
                $video = $metric.Substring(0, $metric.Length - '_metrics.json'.Length) + '_sim2sim.mp4'
                $checked = Test-FullReferenceResult $metric $video "$($spec.Key) adopted full-reference"
                $reference = Join-Path $legacyRefRoot $spec.LegacyReference
                Test-Video $reference $checked.video_frames 1280 720 "$($spec.Key) adopted reference" | Out-Null
                $comparison = Get-UniqueFile $legacyCompareRoot ($spec.LegacyComparison -f $remoteBest.step) "$($spec.Key) adoption comparison"
                Test-Video $comparison $checked.video_frames 2880 600 "$($spec.Key) adopted comparison" | Out-Null
                if ([IO.Path]::GetFileName($comparison) -notmatch 'step(?<old>[0-9]+)_vs_step') { throw "Cannot identify previous step from $comparison" }
                $previousSteps += [int64]$Matches['old']
                $artifacts[$spec.Key] = [ordered]@{ motion_path=$spec.Motion; motion_sha256=(Get-Sha256 $spec.Motion); metrics=$metric; sim_video=$video; reference_video=$reference; comparison_video=$comparison; video_frames=$checked.video_frames }
            }
            if (@($previousSteps | Select-Object -Unique).Count -ne 1) { throw "Adopted comparisons disagree on the previous step." }
            $previousStep = $previousSteps[0]
        } else {
            if ((Get-FreeBytes) -lt 3GB) { throw "Less than 3 GiB free space is available; refusing to start a video run." }
            $previousManifest = Read-Json ([string]$current.manifest_path)
            if ($null -eq $previousManifest -or [string]$previousManifest.status -ne 'complete') { throw "Previous current manifest is missing or incomplete." }
            $previousStep = [int64]$previousManifest.step
            $partialModel = Join-Path $stage "model_step_$($stepText)_kengo.onnx.partial"
            $scp = (Get-Command scp.exe -ErrorAction Stop).Source
            $scpArgs = @('-B','-q','-l',[string]$ScpLimitKbps,'-o','BatchMode=yes','-o','Compression=no','-o',"ConnectTimeout=$SshConnectTimeoutSeconds",'-o','ConnectionAttempts=1','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','StrictHostKeyChecking=yes',("{0}:{1}" -f $RemoteSsh,$remoteOnnx.path),$partialModel)
            Invoke-NativeChecked $scp $scpArgs "Download one ONNX with single-stream SCP" | Out-Null
            if ((Get-Item -LiteralPath $partialModel).Length -ne [int64]$remoteOnnx.bytes -or (Get-Sha256 $partialModel) -ne $remoteOnnx.sha256) { throw "Downloaded ONNX size or SHA does not match the remote artifact." }
            Invoke-PythonJson $script:OnnxValidator @($partialModel, [string]$remoteBest.step) "Validate downloaded ONNX" | Out-Null
            $remoteAfter = Get-RemoteOnnxInfo $remoteBest
            if ($remoteAfter.sha256 -ne $remoteOnnx.sha256 -or $remoteAfter.bytes -ne $remoteOnnx.bytes) { throw "Remote ONNX changed during transfer." }
            $stageModel = Join-Path $stage "model_step_$($stepText)_kengo.onnx"
            [IO.File]::Move($partialModel, $stageModel)
            $ffmpeg = Get-Ffmpeg
            foreach ($spec in $specs) {
                $previousArtifact = Get-Artifact $previousManifest $spec.Key
                $previousChecked = Test-FullReferenceResult ([string]$previousArtifact.metrics) ([string]$previousArtifact.sim_video) "$($spec.Key) previous full-reference"
                $simDir = Join-Path $stage 'sim'; [IO.Directory]::CreateDirectory($simDir) | Out-Null
                $metricsPath = Join-Path $simDir "$($spec.Key).metrics.json"
                $videoPath = Join-Path $simDir "$($spec.Key).sim2sim.mp4"
                $sim = Invoke-NativeChecked $PythonExe @($script:Runner,'--policy',$stageModel,'--motion',$spec.Motion,'--xml',$script:XmlPath,'--headless','--no-real-time','--full-reference','--onnx-threads','1','--progress-interval','10','--metrics-json',$metricsPath,'--video',$videoPath) "Run $($spec.Key) full-reference sim2sim" @(0,1)
                $checked = Test-FullReferenceResult $metricsPath $videoPath "$($spec.Key) new full-reference"
                $expectedExit = if ([bool]$checked.metrics.success) { 0 } else { 1 }
                if ([int]$sim.ExitCode -ne $expectedExit) { throw "$($spec.Key) sim exit code $($sim.ExitCode) disagrees with metrics success=$($checked.metrics.success)." }
                if ($checked.video_frames -ne $previousChecked.video_frames) { throw "$($spec.Key) previous/new frame counts differ." }
                $reference = Ensure-ReferenceVideo $spec $checked.video_frames $previousArtifact
                $compareDir = Join-Path $stage 'compare'; [IO.Directory]::CreateDirectory($compareDir) | Out-Null
                $comparison = Join-Path $compareDir "$($spec.Key).reference_step$($previousStep)_vs_step$($remoteBest.step).mp4"
                New-Comparison $ffmpeg $reference ([string]$previousArtifact.sim_video) $videoPath $checked.video_frames $previousStep $remoteBest.step $comparison
                $finalMetric = Join-Path $final "sim\$($spec.Key).metrics.json"
                $finalVideo = Join-Path $final "sim\$($spec.Key).sim2sim.mp4"
                $finalComparison = Join-Path $final "compare\$([IO.Path]::GetFileName($comparison))"
                $metricsObject = Read-Json $metricsPath; $metricsObject.policy.path=$finalModel; $metricsObject.video.path=$finalVideo; Write-JsonAtomic $metricsObject $metricsPath
                $artifacts[$spec.Key] = [ordered]@{ motion_path=$spec.Motion; motion_sha256=(Get-Sha256 $spec.Motion); metrics=$finalMetric; sim_video=$finalVideo; reference_video=$reference; comparison_video=$finalComparison; video_frames=$checked.video_frames; sim_exit_code=$sim.ExitCode }
            }
        }

        $completed = [DateTime]::UtcNow.ToString('o')
        $manifest = [ordered]@{ schema_version=1; status='complete'; adopted=[bool]$AdoptExisting; provenance=if($null -ne $offlineAdoption){'offline_verified_metadata'}elseif($AdoptExisting){'online_verified_existing'}else{'remote_exported_and_locally_tested'}; adoption_metadata_path=if($null -eq $offlineAdoption){$null}else{$offlineAdoption.source_path}; run_id=$runId; step=[int64]$remoteBest.step; checkpoint_sha256=$remoteBest.checkpoint_sha256; onnx_sha256=$remoteOnnx.sha256; onnx_bytes=$remoteOnnx.bytes; remote_onnx_path=$remoteOnnx.path; model_path=$finalModel; previous_step=$previousStep; previous_run_id=if($null -eq $current){$null}else{$current.run_id}; started_at=$stamp; completed_at=$completed; artifacts=$artifacts; log_path=$script:LogPath }
        Write-JsonAtomic $manifest (Join-Path $stage 'manifest.json')
        $newState = Promote-Run $manifest $stage $final $current
        Invoke-Retention
        Write-Host ("[RESULT_JSON] " + ($manifest | ConvertTo-Json -Depth 20 -Compress))
        Write-Log INFO "Promoted tested step $($newState.step) as current; retention completed."
    }
} catch {
    $exitCode = 1
    if ($script:LogPath) { Write-Log ERROR $_.Exception.Message } else { Write-Error $_.Exception.Message }
    if (-not $CheckOnly) {
        if ($script:ActiveStaging -and [IO.Directory]::Exists($script:ActiveStaging)) { Remove-ManagedDirectory $script:ActiveStaging }
        if ($script:ActivePromotedRun -and [IO.Directory]::Exists($script:ActivePromotedRun)) { Remove-ManagedDirectory $script:ActivePromotedRun }
    }
} finally {
    if ($script:MutexAcquired -and $script:Mutex) { try { $script:Mutex.ReleaseMutex() } catch {} }
    if ($script:Mutex) { $script:Mutex.Dispose() }
}

exit $exitCode
