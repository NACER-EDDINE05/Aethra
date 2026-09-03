# Launches llama.cpp's OpenAI-compatible server (llama-server) with the local
# WhiteRabbitNeo GGUF model so the Aethra backend can talk to it.
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\serve_model.ps1
# Then in another terminal:  python run.py
#
# Interactive model playground (no Aethra needed):  http://localhost:8080

$ErrorActionPreference = "Stop"

$modelPath = "P:\AI MODELS\WhiteRabbitNeo_WhiteRabbitNeo-V3-7B-IQ3_M.gguf"
$alias     = "WhiteRabbitNeo-V3-7B-IQ3_M"
$port      = 8080
$ctxSize   = 2048
# 4 physical threads is as fast as 8 on this CPU and keeps the desktop responsive.
$threads   = [Math]::Min(4, [Environment]::ProcessorCount)

if (-not (Test-Path $modelPath)) { throw "Model file not found: $modelPath" }

# Locate llama-server.exe: PATH -> winget Links -> winget Packages
$llamaServer = @(
    (Get-Command llama-server -ErrorAction SilentlyContinue).Source
    "$env:LOCALAPPDATA\Microsoft\WinGet\Links\llama-server.exe"
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if (-not $llamaServer) {
    $llamaServer = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" `
        -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $llamaServer) {
    throw "llama-server.exe not found. Install it with:  winget install ggml.llamacpp"
}

Write-Host "Server  : $llamaServer"
Write-Host "Model   : $modelPath"
Write-Host "API     : http://localhost:$port/v1  (OpenAI-compatible)"
Write-Host "Web UI  : http://localhost:$port"
Write-Host "Threads : $threads | Context: $ctxSize tokens"
Write-Host "Press Ctrl+C to stop the model server."

& $llamaServer `
    -m $modelPath `
    --alias $alias `
    --host 127.0.0.1 `
    --port $port `
    --ctx-size $ctxSize `
    --threads $threads `
    --parallel 1
