# Minimal chat CLI against the local model server (llama.cpp on :8080).
# Usage:  powershell -ExecutionPolicy Bypass -File .\chat.ps1 "your question"
# Omit the question for an interactive loop.

param([string]$Question = "")

$url = "http://localhost:8080/v1/chat/completions"
$history = @(
    @{ role = "system"; content = "You are Aethra, a concise cybersecurity and DevOps assistant." }
)

function Send-Chat {
    param([string]$Text)
    $script:history += @{ role = "user"; content = $Text }
    $body = @{
        model       = "WhiteRabbitNeo-V3-7B-IQ3_M"
        messages    = $script:history
        temperature = 0.4
        max_tokens  = 400
    } | ConvertTo-Json -Depth 5
    $reply = (Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json" `
        -Body $body -TimeoutSec 900).choices[0].message.content
    $script:history += @{ role = "assistant"; content = $reply }
    Write-Host "`nAethra> $reply`n" -ForegroundColor Green
}

if ($Question) { Send-Chat $Question; return }

Write-Host "Chatting with WhiteRabbitNeo-V3-7B (Ctrl+C to quit). First answer is slow; later ones reuse the prompt cache."
while ($true) {
    $text = Read-Host "You"
    if ($text) { Send-Chat $text }
}
