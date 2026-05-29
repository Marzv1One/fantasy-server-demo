$ErrorActionPreference = 'Stop'

# Read and base64 encode the image
$imagePath = Join-Path $PSScriptRoot 'wallhaven-xezypl-58.png'
$bytes = [System.IO.File]::ReadAllBytes($imagePath)
$b64 = [Convert]::ToBase64String($bytes)
Write-Host "Image size: $($bytes.Length) bytes, base64: $($b64.Length) chars" -ForegroundColor DarkGray

# Build JSON-RPC request
$request = @{
    jsonrpc = '2.0'
    id      = 1
    method  = 'chat.send'
    params  = @{
        prompt = 'Describe this image in detail.'
        files  = @(
            @{
                filename  = 'wallhaven-xezypl-58.png'
                data      = $b64
                mediaType = 'image/png'
            }
        )
    }
} | ConvertTo-Json -Depth 10 -Compress

# Pipe request to server (stdin closes after echo, server exits)
$response = $request | fantasy-server.exe 2>$null

Write-Host "`n--- response ---" -ForegroundColor Green
$response
