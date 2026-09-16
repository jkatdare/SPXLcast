# Deploying SPXLcast to Azure by hand

Run these from the repo root in PowerShell, signed in with `az login`. Names used throughout:
resource group `spxlcast-rg`, region `eastus2`, registry `spxlcastacr`, storage account
`spxlcastsa`, environment `spxlcast-env`, job `spxlcast-daily`, web app `spxlcast-web`.
`infra\deploy.ps1 -DryRun` prints the same sequence; `infra\deploy.ps1` runs it.

Every command below ends in `-o none` (quiet output). Take care not to type `-n none`: a second
`-n` would rename the resource to "none".

## Status as of 2026-09-16

| Step | State |
|---|---|
| 0. Resource providers registered | done (Storage, ContainerRegistry, App, OperationalInsights) |
| 1. Resource group `spxlcast-rg` | done |
| 2. Registry `spxlcastacr` and image build | done |
| 3. Storage account and file share | **not done** (the earlier attempts ran before the provider was registered) |
| 4. Environment `spxlcast-env` | created; **the share is not registered on it yet** |
| 5-7. Job, web app | not created (the job failed for lack of the share; the web spec is fixed below) |

Pick up at step 3.

## 0. Check the subscription and register the resource providers (done)

```powershell
az account show -o table
az provider register -n Microsoft.Storage --wait
az provider register -n Microsoft.ContainerRegistry --wait
az provider register -n Microsoft.App --wait
az provider register -n Microsoft.OperationalInsights --wait
```

A new subscription has none of these registered. Creating a resource before its provider is
registered fails with the misleading message `SubscriptionNotFound`. If `az account show` shows a
subscription you no longer have, run `az account list --refresh -o table` and
`az account set --subscription <id>`.

## 1. Resource group (done)

```powershell
az group create -n spxlcast-rg -l eastus2 -o none
```

## 2. Container registry and cloud image build (done)

```powershell
az acr create -n spxlcastacr -g spxlcast-rg --sku Basic --admin-enabled true -o none
az acr build -r spxlcastacr -t spxlcast:latest . -o none
```

`az acr build` uploads the repo (minus `.dockerignore` entries, so never `.env`) and builds the
Dockerfile in Azure. Re-run it whenever the code changes, or let the GitHub workflow do it.

## 3. Storage account and file share

Storage account names are global and must be 3-24 lowercase letters or digits; add digits if
`spxlcastsa` is taken, and use that name in step 4.

```powershell
az storage account create -n spxlcastsa -g spxlcast-rg -l eastus2 --sku Standard_LRS --kind StorageV2 -o none
az storage share-rm create --storage-account spxlcastsa -g spxlcast-rg -n spxlcast --quota 5 -o none
```

## 4. Register the share on the environment

The environment exists; this attaches the share to it under the name `data`, which both specs
mount at `/data`.

```powershell
$saKey = az storage account keys list -n spxlcastsa -g spxlcast-rg --query "[0].value" -o tsv
az containerapp env storage set -n spxlcast-env -g spxlcast-rg --storage-name data `
    --azure-file-account-name spxlcastsa --azure-file-account-key $saKey `
    --azure-file-share-name spxlcast --access-mode ReadWrite -o none
az containerapp env storage list -n spxlcast-env -g spxlcast-rg -o table
```

The last command must list `data`. (If the environment ever needs recreating:
`az containerapp env create -n spxlcast-env -g spxlcast-rg -l eastus2 -o none`.)

## 5. Fill in the two specs

`infra\job.yaml` and `infra\web.yaml` contain `${...}` placeholders. Render them into your temp
folder; the rendered files contain the registry password and the FRED key, so they must not be
saved into the repo. Re-run this block after any edit to the specs.

```powershell
$fill = @{
  LOCATION     = "eastus2"
  ENV_ID       = az containerapp env show -n spxlcast-env -g spxlcast-rg --query id -o tsv
  JOB_NAME     = "spxlcast-daily"
  WEB_NAME     = "spxlcast-web"
  CRON         = "40 21 * * 1-5"
  ACR_SERVER   = az acr show -n spxlcastacr --query loginServer -o tsv
  ACR_USER     = az acr credential show -n spxlcastacr --query username -o tsv
  ACR_PASSWORD = az acr credential show -n spxlcastacr --query "passwords[0].value" -o tsv
  FRED_API_KEY = ((Get-Content .env | Where-Object { $_ -match "^\s*FRED_API_KEY\s*=" }) -split "=", 2)[1].Trim().Trim("'").Trim('"')
}
foreach ($f in "job", "web") {
  $text = Get-Content "infra\$f.yaml" -Raw
  foreach ($k in $fill.Keys) { $text = $text.Replace('${' + $k + '}', [string]$fill[$k]) }
  Set-Content "$env:TEMP\spxlcast-$f.yaml" $text -Encoding utf8
}
Select-String -Path "$env:TEMP\spxlcast-*.yaml" -Pattern '\$\{'     # must print nothing: no placeholder left
```

The cron expression is UTC: 21:40 UTC is 17:40 in New York in summer and 16:40 in winter, after
the close all year. Weekdays only; on market holidays the job simply logs the previous close,
which the scorer collapses to one row per date.

## 6. Scheduled job

```powershell
az containerapp job create -n spxlcast-daily -g spxlcast-rg --yaml "$env:TEMP\spxlcast-job.yaml" -o none
az containerapp job start -n spxlcast-daily -g spxlcast-rg
az containerapp job execution list -n spxlcast-daily -g spxlcast-rg -o table
az containerapp job logs show -n spxlcast-daily -g spxlcast-rg --container spxlcast-daily
```

"Additional flags were passed along with --yaml" is only a warning about `-o none`; ignore it.
The first run takes a minute or two (it downloads five years of history into the share's cache);
later runs reuse it. The logs end with the one-line rating and the first lines of the score.

## 7. Status page

The web spec sets `ingress.allowInsecure: false` explicitly; without it this CLI version sends
`null` and the API answers "The JSON value could not be converted to System.Boolean".

```powershell
az containerapp create -n spxlcast-web -g spxlcast-rg --yaml "$env:TEMP\spxlcast-web.yaml" -o none
az containerapp show -n spxlcast-web -g spxlcast-rg --query properties.configuration.ingress.fqdn -o tsv
Remove-Item "$env:TEMP\spxlcast-job.yaml", "$env:TEMP\spxlcast-web.yaml"
```

Open `https://<that hostname>`. It scales to zero when idle, so the first request after a pause
takes a few seconds. `/logs/forecast_log.csv` and `/output/forecast.json` are direct downloads.

## 8. GitHub auto-deploy (optional)

`.github/workflows/deploy.yml` runs the tests and, on every push to `master`, builds the image in
the registry and rolls it out to the job and the web app. GitHub authenticates to Azure with a
federated (OIDC) credential, so no password or key is stored anywhere. One-time setup, run once
the resources above exist:

```powershell
$sub    = az account show --query id -o tsv
$tenant = az account show --query tenantId -o tsv
$appId  = az ad app create --display-name spxlcast-github --query appId -o tsv
$objId  = az ad app show --id $appId --query id -o tsv
az ad sp create --id $appId -o none
Start-Sleep 30      # let the new principal propagate before assigning a role
az role assignment create --assignee $appId --role Contributor --scope "/subscriptions/$sub/resourceGroups/spxlcast-rg" -o none
@"
{ "name": "spxlcast-master",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:jkatdare/SPXLcast:ref:refs/heads/master",
  "audiences": ["api://AzureADTokenExchange"] }
"@ | Set-Content "$env:TEMP\fc.json" -Encoding ascii
az ad app federated-credential create --id $objId --parameters "$env:TEMP\fc.json" -o none
gh secret set AZURE_CLIENT_ID --body $appId
gh secret set AZURE_TENANT_ID --body $tenant
gh secret set AZURE_SUBSCRIPTION_ID --body $sub
```

Then push the workflow file and watch it: `gh run watch`. The workflow tags each image with the
commit hash and updates both apps to it, so a rollback is `az containerapp job update ... --image
spxlcastacr.azurecr.io/spxlcast:<older sha>`. Contributor on the resource group is the minimum
that lets it build in the registry and update the apps; it cannot touch anything outside
`spxlcast-rg`.

## Updating and removing

- Code changes: push to `master` (with the workflow), or by hand
  `az acr build -r spxlcastacr -t spxlcast:latest . -o none` followed by
  `az containerapp job update -n spxlcast-daily -g spxlcast-rg --image spxlcastacr.azurecr.io/spxlcast:latest -o none`
  and the same `az containerapp update` for `spxlcast-web`.
- Change the schedule: `az containerapp job update -n spxlcast-daily -g spxlcast-rg --cron-expression "40 21 * * 1-5" -o none`.
- Everything, including the track record on the share: `az group delete -n spxlcast-rg --yes`.
  Download `forecast_log.csv` from the status page first if you want to keep it.

## Cost

The Basic registry is about $5 a month and is the only fixed charge. The job bills only while it
runs (about a minute a day), the web app scales to zero, the 5 GB file share is pennies, and
GitHub Actions minutes on a private repo are within the free monthly allowance for a few runs.
