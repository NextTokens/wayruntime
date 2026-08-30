# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 WayOS Project
#
# realtest.ps1 -- Windows-native end-to-end test against real GGUF
# models, no mocks.  Drives build\win\wayrt.exe exactly as a Windows
# user would.
#
#   powershell -File test\realtest.ps1 [-ModelsDir D:\path\to\models]
#       [-SkipCrossPlatform]
#
# Requires: build\win\wayrt.exe (make WIN=1), the test models, WSL,
# and build\posix\wayrt.  -SkipCrossPlatform is an explicit opt-out
# for a Windows-only run; it does not validate the parity claim.
# NOTE: this file MUST stay pure ASCII -- PowerShell 5.1 misparses
# BOM-less UTF-8, so no unicode characters anywhere in this script.

param(
    [string]$ModelsDir = "D:\Dev\Apps\wayAI\models",
    [switch]$SkipCrossPlatform
)

$ErrorActionPreference = "Continue"
$root  = Split-Path -Parent $PSScriptRoot
$exe   = Join-Path $root "build\win\wayrt.exe"

$stories = Join-Path $ModelsDir "stories15M-q8_0.gguf"
$tinyQ4k = Join-Path $ModelsDir "tiny-q4k-test.gguf"
$gpt2    = Join-Path $ModelsDir "tiny-gpt2-f32.gguf"
$qwen    = Join-Path $ModelsDir "Qwen3-0.6B-Q4_K_M.gguf"

# ---------- preflight (exit 2 = setup problem, not a test failure) ----

if (-not (Test-Path $exe)) {
    Write-Host "realtest-win: $exe missing -- run: make WIN=1"
    exit 2
}
foreach ($m in @($stories, $tinyQ4k, $gpt2, $qwen)) {
    if (-not (Test-Path $m)) {
        Write-Host "realtest-win: model missing: $m (use -ModelsDir)"
        exit 2
    }
}

$wsl = $null
$repoWsl = ""
$storiesWsl = ""
$posixExe = Join-Path $root "build\posix\wayrt"
if (-not $SkipCrossPlatform) {
    $wsl = Get-Command wsl -ErrorAction SilentlyContinue
    if (-not $wsl) {
        Write-Host "realtest-win: WSL missing; parity test is required by default"
        Write-Host "realtest-win: install/enable WSL or intentionally pass -SkipCrossPlatform"
        exit 2
    }
    if (-not (Test-Path $posixExe -PathType Leaf)) {
        Write-Host "realtest-win: $posixExe missing; parity test is required by default"
        Write-Host "realtest-win: build it in WSL or intentionally pass -SkipCrossPlatform"
        exit 2
    }

    $repoWsl = (& $wsl.Source -e wslpath -a ($root -replace "\\", "/") `
        2>$null) -join ""
    $repoWslRc = $LASTEXITCODE
    $storiesWsl = (& $wsl.Source -e wslpath -a ($stories -replace "\\", "/") `
        2>$null) -join ""
    $storiesWslRc = $LASTEXITCODE
    $repoWsl = $repoWsl.Trim()
    $storiesWsl = $storiesWsl.Trim()
    if ($repoWslRc -ne 0 -or $storiesWslRc -ne 0 -or
            [string]::IsNullOrWhiteSpace($repoWsl) -or
            [string]::IsNullOrWhiteSpace($storiesWsl)) {
        Write-Host "realtest-win: WSL path conversion failed; parity test cannot run"
        Write-Host "realtest-win: intentionally pass -SkipCrossPlatform for a Windows-only run"
        exit 2
    }

    & $wsl.Source -e test -x "$repoWsl/build/posix/wayrt"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "realtest-win: build/posix/wayrt is not executable inside WSL"
        Write-Host "realtest-win: rebuild it in WSL or intentionally pass -SkipCrossPlatform"
        exit 2
    }
}

$T = Join-Path $env:TEMP "wayrt-realtest"
New-Item -ItemType Directory -Force $T | Out-Null

$script:pass = 0
$script:fail = 0
$script:step = 0
function Check($cond, $msg, $detail) {
    $script:step++
    if ($cond) {
        $script:pass++
        Write-Host ("{0,2}. ok   {1}" -f $script:step, $msg)
    } else {
        $script:fail++
        Write-Host ("{0,2}. FAIL {1}" -f $script:step, $msg) -ForegroundColor Red
        if ($detail) { Write-Host ("      " + $detail) }
    }
}

# ProcessStartInfo accepts one Windows command-line string on PowerShell
# 5.1. Quote one argv element with the CommandLineToArgvW backslash rules.
function Quote-NativeArg([string]$value) {
    if ($null -eq $value -or $value.Length -eq 0) { return '""' }
    if ($value -notmatch '[\s"]') { return $value }

    $quoted = New-Object System.Text.StringBuilder
    [void]$quoted.Append('"')
    $slashes = 0
    foreach ($ch in $value.ToCharArray()) {
        if ($ch -eq '\') {
            $slashes++
        } elseif ($ch -eq '"') {
            if ($slashes -gt 0) {
                [void]$quoted.Append((('\' * ($slashes * 2)) -join ''))
            }
            [void]$quoted.Append('\"')
            $slashes = 0
        } else {
            if ($slashes -gt 0) {
                [void]$quoted.Append((('\' * $slashes) -join ''))
                $slashes = 0
            }
            [void]$quoted.Append($ch)
        }
    }
    if ($slashes -gt 0) {
        [void]$quoted.Append((('\' * ($slashes * 2)) -join ''))
    }
    [void]$quoted.Append('"')
    return $quoted.ToString()
}

# Capture stdout from a native process without sending it through the
# PowerShell text pipeline. StandardOutput.BaseStream preserves the exact
# bytes emitted by both wayrt.exe and wsl.exe.
function Invoke-NativeCapture($filePath, [string[]]$argumentList,
        $stdoutPath, $stderrPath) {
    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $filePath
    $start.Arguments = (($argumentList | ForEach-Object {
        Quote-NativeArg $_
    }) -join ' ')
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $start
    try {
        [void]$process.Start()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $stdout = [System.IO.File]::Create($stdoutPath)
        try {
            $process.StandardOutput.BaseStream.CopyTo($stdout)
        } finally {
            $stdout.Dispose()
        }
        $process.WaitForExit()
        $stderrText = $stderrTask.GetAwaiter().GetResult()
        [System.IO.File]::WriteAllText($stderrPath, $stderrText)
        return $process.ExitCode
    } catch {
        [System.IO.File]::WriteAllText($stderrPath, $_.Exception.Message)
        return -1
    } finally {
        $process.Dispose()
    }
}

function Test-FilesByteEqual($leftPath, $rightPath) {
    $left = [System.IO.File]::ReadAllBytes($leftPath)
    $right = [System.IO.File]::ReadAllBytes($rightPath)
    if ($left.Length -ne $right.Length) { return $false }
    for ($i = 0; $i -lt $left.Length; $i++) {
        if ($left[$i] -ne $right[$i]) { return $false }
    }
    return $true
}

# ---------- 1-2. verify on supported models ---------------------------

$out = (& $exe verify $stories 2>"$T\err1.txt") -join "`n"
Check ($LASTEXITCODE -eq 0 -and $out -match "arch:\s+llama") `
    "verify stories15M-q8_0 (exit 0, arch llama)" "exit=$LASTEXITCODE"

$out = (& $exe verify $tinyQ4k 2>"$T\err2.txt") -join "`n"
Check ($LASTEXITCODE -eq 0 -and $out -match "arch:\s+llama") `
    "verify tiny-q4k-test (exit 0, arch llama)" "exit=$LASTEXITCODE"

# ---------- 3. unsupported architecture is refused --------------------

& $exe verify $gpt2 1>"$T\gpt2.out" 2>"$T\gpt2.err"
$rc = $LASTEXITCODE
$errTxt = ""
if (Test-Path "$T\gpt2.err") { $errTxt = (Get-Content "$T\gpt2.err") -join "`n" }
Check ($rc -eq 2 -and $errTxt -match "gpt2") `
    "tiny-gpt2-f32 refused (exit exactly 2, stderr names the arch)" `
    "exit=$rc"

# ---------- 4. greedy determinism (2 runs byte-identical) -------------

$g1 = (& $exe generate --raw --greedy --max-tokens 24 `
        --prompt "Once upon a time" $stories 2>$null) -join "`n"
$rc1 = $LASTEXITCODE
$g2 = (& $exe generate --raw --greedy --max-tokens 24 `
        --prompt "Once upon a time" $stories 2>$null) -join "`n"
Check ($rc1 -eq 0 -and $LASTEXITCODE -eq 0 -and $g1.Length -gt 0 -and $g1 -ceq $g2) `
    "generate --raw --greedy deterministic on stories15M (2 runs identical)" `
    "exit=$rc1/$LASTEXITCODE"

# ---------- 4b. cross-platform: byte-identical vs the Linux build -----
# The documented bar (docs/TESTING.md layer 5) is byte-identical greedy
# output to the POSIX build for the same model and prompt.  It is a
# required hard step unless the caller explicitly requests Windows-only.

if (-not $SkipCrossPlatform) {
    $winParityOut = Join-Path $T "parity-win.out"
    $winParityErr = Join-Path $T "parity-win.err"
    $linParityOut = Join-Path $T "parity-linux.out"
    $linParityErr = Join-Path $T "parity-linux.err"
    $generateArgs = @("generate", "--raw", "--greedy", "--max-tokens", "24",
        "--prompt", "Once upon a time", $stories)
    $linGenerateArgs = @("-e", "$repoWsl/build/posix/wayrt", "generate",
        "--raw", "--greedy", "--max-tokens", "24", "--prompt",
        "Once upon a time", $storiesWsl)

    $winParityRc = Invoke-NativeCapture $exe $generateArgs `
        $winParityOut $winParityErr
    $linParityRc = Invoke-NativeCapture $wsl.Source $linGenerateArgs `
        $linParityOut $linParityErr
    $winParityLen = (Get-Item $winParityOut).Length
    $linParityLen = (Get-Item $linParityOut).Length
    $winParityHash = (Get-FileHash -Algorithm SHA256 $winParityOut).Hash
    $linParityHash = (Get-FileHash -Algorithm SHA256 $linParityOut).Hash
    $parityEqual = Test-FilesByteEqual $winParityOut $linParityOut
    Check ($winParityRc -eq 0 -and $linParityRc -eq 0 -and
            $winParityLen -gt 0 -and $parityEqual) `
        "cross-platform greedy output byte-identical (Windows vs Linux build)" `
        ("windows-exit=$winParityRc linux-exit=$linParityRc " +
         "windows-bytes=$winParityLen linux-bytes=$linParityLen " +
         "windows-sha256=$winParityHash linux-sha256=$linParityHash")
} else {
    Write-Host "note: cross-platform parity intentionally NOT RUN (-SkipCrossPlatform)"
}

# ---------- 5. bench: counters + throughput ---------------------------

$bench = & $exe bench --tokens 8 $stories 2>$null
$rc = $LASTEXITCODE
$benchArr = @($bench)
$ci = [array]::IndexOf($benchArr, "counters:")
$nCounters = 0
if ($ci -ge 0) {
    $nCounters = @($benchArr[($ci + 1)..($benchArr.Count - 1)] |
        Where-Object { $_ -match "^  \w+\s+\d+" }).Count
}
$tokps = 0.0
$decodeLine = $benchArr | Where-Object { $_ -match "^decode:" }
if ($decodeLine -match "([0-9.]+) tok/s") { $tokps = [double]$Matches[1] }
Check ($rc -eq 0 -and $nCounters -eq 12 -and $tokps -gt 0) `
    "bench --tokens 8 (exit 0, 12 counters, nonzero tok/s)" `
    "exit=$rc counters=$nCounters tokps=$tokps"

# ---------- 6. Qwen3 chat template answers 2+2 ------------------------

# Chat template auto-detected (no --raw); greedy default.  Assert only
# that the answer contains 4 -- never an exact token stream, so
# tokenizer-fidelity work cannot break this test.
$ans = (& $exe generate --max-tokens 48 --prompt "What is 2+2?" $qwen 2>$null) -join "`n"
Check ($LASTEXITCODE -eq 0 -and $ans -match "4") `
    "Qwen3 chat-template answer to 2+2 contains 4" `
    ("exit=$LASTEXITCODE answer=" + $ans.Substring(0, [Math]::Min(80, $ans.Length)))

# ---------- summary ---------------------------------------------------

$total = $script:pass + $script:fail
Write-Host ""
if ($SkipCrossPlatform) {
    Write-Host "realtest-win: $($script:pass)/$total Windows-only checks passed"
    Write-Host "realtest-win: cross-platform parity NOT RUN (-SkipCrossPlatform)"
} else {
    Write-Host "realtest-win: $($script:pass)/$total passed (parity included)"
}
exit $(if ($script:fail -eq 0) { 0 } else { 1 })
