# Deploying SPXLcast to Azure by hand

Run these from the repo root in PowerShell, signed in with `az login`. Names used throughout:
resource group `spxlcast-rg`, region `eastus2`, registry `spxlcastacr`, storage account
`spxlcastsa`, environment `spxlcast-env`, job `spxlcast-daily`, web app `spxlcast-web`.
`infra\deploy.ps1 -DryRun` prints the same sequence; `infra\deploy.ps1` runs it.

Every command below ends in `-o none` (quiet output). Take care not to type `-n none`: a second
`-n` would rename the resource to "none".

## Status as of 2026-09-16

Steps 0 to 7 are done and the pipeline works: the job ran at 22:13 UTC, logged a close row for
2026-09-16 rating HOLD, and the page serves it at
`https://spxlcast-web.redplant-3805e781.eastus2.azurecontainerapps.io`.

Remaining: step 8 (point `spxlcast.com` at it) and step 9 (GitHub auto-deploy, optional). The
earlier steps are kept below as the record of how it was built and for rebuilding from scratch.

Two problems hit on the way, both fixed here: creating a resource before its provider is
registered fails with the misleading `SubscriptionNotFound`, and the web spec must set
`ingress.allowInsecure: false` explicitly or the CLI sends `null` and the API rejects it.

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
  CRON         = "40 13-21 * * 1-5"
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

The cron expression is UTC and runs hourly at :40 from 13:40 to 21:40, weekdays. In summer that is
9:40 to 17:40 New York time, in winter 8:40 to 16:40, so it always covers the session and ends with
one run after the close. Intraday runs log live quotes (marked `intraday`); the after-close run logs
the close, and the scorer keeps that one row per date. While the session is open the price cache
expires after 30 minutes so each hourly run sees a fresh quote. On market holidays the job simply
logs the previous close.

The GitHub workflow re-applies this cron expression (and the retry limit) on every deploy, so a
schedule changed by hand is overwritten by the next push; change `CRON` in
`.github/workflows/deploy.yml` instead.

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

Open `https://<that hostname>`. The app is always on (one replica): besides serving the page it
runs the live loop (`serve --live`, also switched on by `SPXLCAST_LIVE=1`, which the image sets),
which every minute of the regular session fetches the SPXL quote, restates the last full forecast
at that price into `output/live.json` and appends a row to `logs/spot_log.csv`. The page shows
that block at the top and refreshes itself every minute. `/logs/forecast_log.csv`,
`/logs/spot_log.csv`, `/output/forecast.json` and `/output/live.json` are direct downloads.

To switch an existing deployment from scale-to-zero to always on without re-running the whole
script (the workflow also does this on every deploy):

```powershell
az containerapp update -n spxlcast-web -g spxlcast-rg --min-replicas 1 --max-replicas 1 --set-env-vars SPXLCAST_LIVE=1 -o none
az containerapp job update -n spxlcast-daily -g spxlcast-rg --cron-expression "40 13-21 * * 1-5" --replica-retry-limit 2 -o none
```

## 8. Custom domain (spxlcast.com, registered at Cloudflare)

The bare domain cannot use a CNAME, so it points at the environment's static IP with an A record.
Get the two values the records need (they are stable unless the app or environment is recreated):

```powershell
az containerapp show -n spxlcast-web -g spxlcast-rg --query properties.customDomainVerificationId -o tsv
az containerapp env show -n spxlcast-env -g spxlcast-rg --query properties.staticIp -o tsv
```

In the Cloudflare dashboard, under the domain's DNS records, add:

| Type | Name | Content | Proxy status |
|---|---|---|---|
| A | `@` | the static IP (57.162.34.53) | **DNS only** |
| TXT | `asuid` | the verification ID | n/a |

`DNS only` (a grey cloud, not orange) is required. A proxied record makes Cloudflare answer the
request itself, so Azure can never verify ownership and the certificate never issues.

Those two records are all the DNS an apex domain needs. There is **no `_acme-challenge` record**
in this process; that is an ACME convention Container Apps does not use here.

Attach the hostname in **two steps**. `bind` on its own fails with
`RequireCustomHostnameInEnvironment`, because it tries to create the certificate before the
hostname exists on the app; `add` registers it first, then `bind` issues the certificate.

**The validation method depends on the record type, and getting it wrong wastes certificates:**

| Domain type | Record | `--validation-method` |
|---|---|---|
| Apex (`spxlcast.com`) | A | `HTTP` |
| Subdomain (`www.spxlcast.com`) | CNAME | `CNAME` |

`TXT` is a valid flag value but is not the documented path for either case here, and a certificate
created with it sits in `Pending` forever.

```powershell
az containerapp hostname add -n spxlcast-web -g spxlcast-rg --hostname spxlcast.com
az containerapp hostname bind -n spxlcast-web -g spxlcast-rg --hostname spxlcast.com `
    --environment spxlcast-env --validation-method HTTP
az containerapp hostname list -n spxlcast-web -g spxlcast-rg -o table
```

`BindingType` reads `Disabled` after `add` and `SniEnabled` once the certificate is bound.
Issuance takes a few minutes. Until then the site answers on the `azurecontainerapps.io` hostname
as before, so nothing is down while you wait.

If a certificate gets stuck in `Pending`, delete it and bind again rather than re-running `bind`,
which only re-reads the stuck one:

```powershell
az containerapp env certificate list -n spxlcast-env -g spxlcast-rg --managed-certificates-only `
    --query "[].{name:name, domain:properties.subjectName, state:properties.provisioningState}" -o table
az containerapp env certificate delete -n spxlcast-env -g spxlcast-rg --certificate <NAME> --yes
```

Microsoft's documented requirements for the free certificate, all of which must hold at renewal
time too, not just at issuance:

- HTTP ingress enabled and the app publicly reachable from DigiCert's validation addresses.
- Apex domains use an A record to the environment IP; subdomains use a CNAME **directly** to the
  app's generated hostname. An intermediate hop blocks issuance, and Microsoft names Cloudflare
  specifically, which is why the record must be `DNS only` rather than proxied.
- If the root domain has any CAA record, it must include `0 issue digicert.com`. No CAA record at
  all is fine. Check with `curl -s "https://dns.google/resolve?name=spxlcast.com&type=CAA"`.
- The app must stay running, since a stopped app cannot answer the validation request.

### Also serving www.spxlcast.com

A subdomain can use a CNAME, so `www` points at the app's hostname rather than at an IP and never
needs editing if the environment's address changes. Add two more Cloudflare records:

| Type | Name | Content | Proxy status |
|---|---|---|---|
| CNAME | `www` | `spxlcast-web.redplant-3805e781.eastus2.azurecontainerapps.io` | **DNS only** |
| TXT | `asuid.www` | the same verification ID as the apex | n/a |

The verification ID belongs to the app, not to the hostname, so both TXT records carry the same
value. Then add and bind it, using CNAME validation this time. CNAME validation checks the record
above, so unlike the apex this needs no `_acme-challenge` record:

```powershell
az containerapp hostname add -n spxlcast-web -g spxlcast-rg --hostname www.spxlcast.com
az containerapp hostname bind -n spxlcast-web -g spxlcast-rg --hostname www.spxlcast.com `
    --environment spxlcast-env --validation-method CNAME
az containerapp hostname list -n spxlcast-web -g spxlcast-rg -o table
```

Both names now serve the page, each with its own free managed certificate. If you would rather
have one canonical address, skip the binding and instead set the `www` record to Proxied (orange)
with a Cloudflare redirect rule sending `www.spxlcast.com` to `spxlcast.com`. That needs no Azure
work and no second certificate, but the redirect is configured in Cloudflare rather than here.

## 9. GitHub auto-deploy (optional)

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
- Change the schedule: edit `CRON` in `.github/workflows/deploy.yml` and push (or, until the next
  push, `az containerapp job update -n spxlcast-daily -g spxlcast-rg --cron-expression "..." -o none`).
- Is it alive? `https://spxlcast.com/output/live.json` should be under two minutes old during the
  session; `az containerapp logs show -n spxlcast-web -g spxlcast-rg --tail 50` shows the loop, and
  `az containerapp job execution list -n spxlcast-daily -g spxlcast-rg -o table` the hourly runs.
- Everything, including the track record on the share: `az group delete -n spxlcast-rg --yes`.
  Download `forecast_log.csv` from the status page first if you want to keep it.

## Cost

The Basic registry is about $5 a month. The always-on web app (0.25 vCPU, 0.5 GiB) is the other
fixed charge: roughly $14 a month at consumption prices after the free monthly grant. The job bills
only while it runs (nine one-minute runs a day, under $1 a month), the 5 GB file share is pennies,
and GitHub Actions minutes on a private repo are within the free monthly allowance for a few runs.
