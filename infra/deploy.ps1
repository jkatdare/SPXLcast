<#
Deploy SPXLcast to Azure: a scheduled Container Apps Job runs the forecast hourly through the US session
(plus once after the close) and appends to the track record on an Azure Files share; an always-on Container App
serves a status page and re-prices the forecast at the live quote every minute of the session.

Resources (one resource group):
  1. Azure Container Registry (Basic)          builds and stores the image (~$5/month, the main cost)
  2. Storage account + file share "spxlcast"   persistent logs, outputs and data cache (pennies)
  3. Container Apps environment (Consumption)  free to exist
  4. Container Apps Job    <prefix>-daily      infra/job.yaml, cron 40 13-21 * * 1-5 (UTC), hourly through the session
  5. Container App         <prefix>-web        infra/web.yaml, external HTTPS, always on (1 replica)

Usage (PowerShell, from the repo root, after `az login`):
  .\infra\deploy.ps1 -DryRun                      # print every command, change nothing
  .\infra\deploy.ps1                              # eastus2, prefix spxlcast
  .\infra\deploy.ps1 -Prefix mycast -Location westus2
Re-running rebuilds the image and updates the job and web app in place.
#>
param(
    [string]$Prefix = "spxlcast",
    [string]$Location = "eastus2",
    [string]$ResourceGroup = "$Prefix-rg",
    [string]$Cron = "40 13-21 * * 1-5",
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"

# Plain functions (no param block) so tokens like -n, -g and -o reach az untouched.
function Run {
    # Print the az command (secrets masked), then run it unless -DryRun.
    $azArgs = @($args)
    $shown = ($azArgs | ForEach-Object { if ("$_" -match "^(fred-key=|.*password.*=).+" -or "$_" -match "^[A-Za-z0-9+/]{40,}={0,2}$") { "<secret>" } else { "$_" } }) -join " "
    Write-Host "az $shown" -ForegroundColor Cyan
    if (-not $DryRun) { az @azArgs; if ($LASTEXITCODE -ne 0) { throw "az command failed" } }
}
function Query {
    # az query that must return a value even in dry-run mode (a placeholder then).
    $placeholder = $args[0]; $azArgs = @($args[1..($args.Count - 1)])
    if ($DryRun) { Write-Host "az $($azArgs -join ' ')" -ForegroundColor DarkGray; return $placeholder }
    $out = az @azArgs; if ($LASTEXITCODE -ne 0) { throw "az command failed" }; return $out
}
function Exists {
    # Does the resource exist? (always "no" in dry-run mode, so the create path is shown)
    $azArgs = @($args)
    if ($DryRun) { return $false }
    # Function scope: under "Stop", Windows PowerShell 5.1 turns az's "not found" on stderr into a terminating error.
    $ErrorActionPreference = "Continue"
    $null = az @azArgs -o none 2>&1
    return ($LASTEXITCODE -eq 0)
}
function GitOut {
    # git output, or nothing when git fails (same stderr caveat as Exists)
    $ErrorActionPreference = "Continue"
    $out = git @args 2>$null
    if ($LASTEXITCODE -eq 0) { return $out }
}

# ---- FRED key from .env (never stored in the repo) ----------------------------------------
$fredKey = $env:FRED_API_KEY
if (-not $fredKey -and (Test-Path ".env")) {
    $line = Get-Content ".env" | Where-Object { $_ -match "^\s*FRED_API_KEY\s*=" } | Select-Object -First 1
    if ($line) { $fredKey = ($line -split "=", 2)[1].Trim().Trim("'").Trim('"') }
}
if (-not $fredKey) { Write-Warning "No FRED_API_KEY found in the environment or .env; the job will run without FRED."; $fredKey = "" }

$share = "spxlcast"; $envName = "$Prefix-env"; $jobName = "$Prefix-daily"; $webName = "$Prefix-web"; $image = "spxlcast:latest"
# Registry and storage names must be globally unique and alphanumeric: reuse existing ones if present.
$acr = Query "$($Prefix)acr" acr list -g $ResourceGroup --query "[0].name" -o tsv
if (-not $acr) { $acr = ($Prefix + "acr" + (Get-Random -Minimum 1000 -Maximum 9999)).ToLower() -replace "[^a-z0-9]", "" }
$storage = Query "$($Prefix)sa" storage account list -g $ResourceGroup --query "[0].name" -o tsv
if (-not $storage) { $storage = ($Prefix + "sa" + (Get-Random -Minimum 10000 -Maximum 99999)).ToLower() -replace "[^a-z0-9]", "" }

Write-Host "`n== 1. resource group ==" -ForegroundColor Yellow
Run group create -n $ResourceGroup -l $Location -o none

Write-Host "`n== 2. container registry and cloud image build (no local Docker needed) ==" -ForegroundColor Yellow
Run provider register -n Microsoft.ContainerRegistry --wait -o none
Run acr create -n $acr -g $ResourceGroup --sku Basic --admin-enabled true -o none
$build = "unknown"   # the git commit, logged with every forecast
$sha = GitOut rev-parse --short=12 HEAD
if ($sha) {
    $build = "$sha".Trim()
    # acr build uploads the working tree, so uncommitted or untracked code must not ship under a clean id.
    # The build column keeps 12 characters: shorten the sha so "-dirty" survives.
    if (GitOut status --porcelain --untracked-files=normal spxlcast scripts infra/daily.sh Dockerfile requirements.txt pyproject.toml) {
        $build = $build.Substring(0, 6) + "-dirty"
    }
}
Run acr build -r $acr -t $image --build-arg "SPXLCAST_BUILD=$build" . -o none
$acrServer = Query "$acr.azurecr.io" acr show -n $acr --query loginServer -o tsv
$acrUser   = Query $acr acr credential show -n $acr --query username -o tsv
$acrPass   = Query "<acr-password>" acr credential show -n $acr --query "passwords[0].value" -o tsv

Write-Host "`n== 3. storage account and file share (persistent /data) ==" -ForegroundColor Yellow
# A fresh subscription may not have the provider registered; Azure then reports "SubscriptionNotFound".
Run provider register -n Microsoft.Storage --wait -o none
Run storage account create -n $storage -g $ResourceGroup -l $Location --sku Standard_LRS --kind StorageV2 -o none
$saKey = Query "<storage-key>" storage account keys list -n $storage -g $ResourceGroup --query "[0].value" -o tsv
Run storage share-rm create --storage-account $storage -g $ResourceGroup -n $share --quota 5 -o none

Write-Host "`n== 4. container apps environment, with the share mounted as storage 'data' ==" -ForegroundColor Yellow
Run extension add --name containerapp --upgrade -o none
Run provider register -n Microsoft.App --wait -o none
Run provider register -n Microsoft.OperationalInsights --wait -o none
Run containerapp env create -n $envName -g $ResourceGroup -l $Location -o none
Run containerapp env storage set -n $envName -g $ResourceGroup --storage-name data `
    --azure-file-account-name $storage --azure-file-account-key $saKey --azure-file-share-name $share --access-mode ReadWrite -o none
$envId = Query "<environment-id>" containerapp env show -n $envName -g $ResourceGroup --query id -o tsv

# Render the two YAML specs from the templates.
$fill = @{ LOCATION = $Location; ENV_ID = $envId; JOB_NAME = $jobName; WEB_NAME = $webName; CRON = $Cron
           ACR_SERVER = $acrServer; ACR_USER = $acrUser; ACR_PASSWORD = $acrPass; FRED_API_KEY = $fredKey }
function Render([string]$template, [string]$target) {
    $text = Get-Content $template -Raw
    foreach ($k in $fill.Keys) { $text = $text.Replace('${' + $k + '}', [string]$fill[$k]) }
    if ($text -match '\$\{[A-Z0-9_]+\}') { throw "$template has a placeholder that deploy.ps1 does not fill: $($Matches[0])" }
    Set-Content -Path $target -Value $text -Encoding utf8
    $masked = $text.Replace($acrPass, "<acr-password>"); if ($fredKey) { $masked = $masked.Replace($fredKey, "<fred-key>") }
    Write-Host "---- $target ----" -ForegroundColor DarkGray; Write-Host $masked
}
$jobSpec = Join-Path $env:TEMP "spxlcast-job.yaml"; $webSpec = Join-Path $env:TEMP "spxlcast-web.yaml"
try {
    Render "infra/job.yaml" $jobSpec
    Render "infra/web.yaml" $webSpec

    Write-Host "`n== 5. scheduled job ($Cron UTC) ==" -ForegroundColor Yellow
    $jobExists = Exists containerapp job show -n $jobName -g $ResourceGroup
    if ($jobExists) { Run containerapp job update -n $jobName -g $ResourceGroup --yaml $jobSpec -o none }
    else            { Run containerapp job create -n $jobName -g $ResourceGroup --yaml $jobSpec -o none }

    Write-Host "`n== 6. status page web app ==" -ForegroundColor Yellow
    $webExists = Exists containerapp show -n $webName -g $ResourceGroup
    if ($webExists) { Run containerapp update -n $webName -g $ResourceGroup --yaml $webSpec -o none }
    else            { Run containerapp create -n $webName -g $ResourceGroup --yaml $webSpec -o none }
} finally {
    Remove-Item $jobSpec, $webSpec -ErrorAction SilentlyContinue   # rendered specs contain secrets, whatever happened
}

$fqdn = Query "<web-fqdn>" containerapp show -n $webName -g $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
Write-Host ""
Write-Host "Status page:      https://$fqdn"
Write-Host "Run the job now:  az containerapp job start -n $jobName -g $ResourceGroup"
Write-Host "Job history:      az containerapp job execution list -n $jobName -g $ResourceGroup -o table"
Write-Host "Job logs:         az containerapp job logs show -n $jobName -g $ResourceGroup --container $jobName"
Write-Host "Tear down:        az group delete -n $ResourceGroup --yes"
