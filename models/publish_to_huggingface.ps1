$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    throw "Hugging Face CLI is not installed. Install huggingface_hub, then run this script again."
}

hf auth whoami | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Hugging Face CLI is not authenticated. Run 'hf auth login' using the official login flow, then rerun this script."
}

hf upload prajwal90/AirQuality-SLM-Scratch-Qwen `
    ".\models\model-a-airquality-classifier" `
    "two-model-package/model-a-airquality-classifier" `
    --repo-type model `
    --commit-message "Add organized Model A package"

hf upload prajwal90/AirQuality-SLM-Conversational-Qwen2.5-1.5B `
    ".\models\model-b-conversational" `
    "." `
    --repo-type model `
    --commit-message "Publish Model B conversational configuration"

Write-Host "Both Hugging Face model packages were uploaded successfully."
