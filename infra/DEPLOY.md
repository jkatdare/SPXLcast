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

Update 2026-09-26: `https://spxlcast.com` is live. `www.spxlcast.com` is **not**: its Cloudflare
records exist, but the hostname was never added and bound on the app, so `https://www.spxlcast.com`
fails the TLS handshake and `http://www.spxlcast.com` returns Azure's 404. The `hostname add` and
`hostname bind` commands under "Also serving www.spxlcast.com" (step 8) finish it.

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
$build = git rev-parse --short=12 HEAD
if (git status --porcelain --untracked-files=normal spxlcast scripts infra/daily.sh Dockerfile requirements.txt pyproject.toml) { $build = $build.Substring(0, 6) + "-dirty" }
az acr build -r spxlcastacr -t spxlcast:latest --build-arg "SPXLCAST_BUILD=$build" . -o none
```

`az acr build` uploads the working tree (minus `.dockerignore` entries, so never a `.env` at any
depth) and builds the Dockerfile in Azure. Re-run it whenever the code changes, or let the GitHub
workflow do it. The build argument stamps the image with the git commit, which every logged
forecast records in its `build` column (without it the column says `unknown`). Uncommitted or
untracked changes to the shipped files are uploaded too, so the stamp then reads `<sha>-dirty`
(the sha cut to 6 characters, since the column keeps 12).

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
Select-String -Path "$env:TEMP\spxlcast-*.yaml" -Pattern '\$\{[A-Z0-9_]+\}'     # must print nothing: no placeholder left
```

The cron expression is UTC and runs hourly at :40 from 13:40 to 21:40, weekdays. In summer that is
9:40 to 17:40 New York time, in winter 8:40 to 16:40, so it always covers the session and ends with
one run after the close. Intraday runs log live quotes (marked `intraday`); the after-close run logs
the close, and the scorer keeps that one row per date. While the session is open the price cache
expires after 30 minutes so each hourly run sees a fresh quote. On market holidays the job simply
logs the previous close, and on 13:00 early-close days the runs after 13:00 ET log the close. In
winter the 13:40 UTC run is before the open and also logs the previous close; the scorer ignores it
in favour of the run made on the day.

The GitHub workflow re-applies this cron expression (and the retry limit) on every deploy, so a
schedule changed by hand is overwritten by the next push; change `CRON` in
`.github/workflows/deploy.yml` instead, and `SCHEDULE` in `spxlcast/health.py` with it (a test
fails while they differ).

## 6. Scheduled job

```powershell
az containerapp job create -n spxlcast-daily -g spxlcast-rg --yaml "$env:TEMP\spxlcast-job.yaml" -o none
az containerapp job start -n spxlcast-daily -g spxlcast-rg
az containerapp job execution list -n spxlcast-daily -g spxlcast-rg -o table
az containerapp job logs show -n spxlcast-daily -g spxlcast-rg --container spxlcast-daily
```

"Additional flags were passed along with --yaml" is only a warning about `-o none`; ignore it.
The first run takes a minute or two (it downloads five years of history into the share's cache);
later runs reuse it. The logs end with any `warning:` line of the report, the report's last lines
(what the run appended, archived and wrote) and the first lines of the score.

The page's report and track record are replaced only when their step succeeds. A failed forecast
fails the execution and leaves the previous report up; its output is in the logs and in
`last_error.txt` at the root of the share, which the page does not serve. A failed score step
(including a price download that returns nothing) keeps the previous track record, does not fail
the execution (a retry would re-run the forecast) and leaves `output/score_error.txt`, which the
health check reports.

`forecast.json` and `fan.png` are replaced atomically (a temporary file, then a rename), so the live
loop and downloads never see a partial file, and the forecast command writes them before it renders
the report. If rendering fails, `report.txt` holds a `warning: the report could not be rendered in
full: ...` line after the sections that did render (the job log repeats it), but the command still
exits 0 with the log row, the archive, `forecast.json` and `fan.png` all written; run it with `-v`
for the full traceback. If the log row cannot be appended, the outputs are still written and the
command exits 1, so the execution fails and is retried.

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
that block at the top and refreshes itself every minute during the session (every five minutes
otherwise). Outside the session the loop fetches no quotes. After the close it marks `live.json`
closed at the last quote. After each full run outside the session (the after-close runs, and the
winter pre-open run) it restates `live.json` at that run's own price, so the block always matches
the report below it. `/logs/forecast_log.csv`, `/logs/spot_log.csv`, `/output/forecast.json` and
`/output/live.json` are direct downloads. Only files inside `output/` and `logs/` are served (no
directory listings; HEAD answers like GET).

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
above, so unlike the apex this needs no `_acme-challenge` record.

**Not done yet (2026-09-26):** both records above are in Cloudflare, but `hostname add` and
`hostname bind` below have not been run, so `www.spxlcast.com` does not work: HTTPS resets during
the handshake (the app has no binding or certificate for that name) and HTTP gets Azure's 404. Run
them to finish it:

```powershell
az containerapp hostname add -n spxlcast-web -g spxlcast-rg --hostname www.spxlcast.com
az containerapp hostname bind -n spxlcast-web -g spxlcast-rg --hostname www.spxlcast.com `
    --environment spxlcast-env --validation-method CNAME
az containerapp hostname list -n spxlcast-web -g spxlcast-rg -o table
```

Once `hostname list` shows `www.spxlcast.com` as `SniEnabled`, both names serve the page, each
with its own free managed certificate. `infra/web.yaml` lists no custom domains, and `deploy.ps1`
re-applies it to an existing app (`containerapp update --yaml`; the GitHub workflow only swaps the
image): after re-running the script, check `hostname list` and bind again if a name is gone. If you
would rather have one canonical address, skip the binding and instead set the `www` record to
Proxied (orange) with a Cloudflare redirect rule sending `www.spxlcast.com` to `spxlcast.com`. That
needs no Azure work and no second certificate, but the redirect is configured in Cloudflare rather
than here. If `www` is not wanted at all, delete its CNAME and `asuid.www` TXT records instead.

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
# The subject must match the token GitHub presents *exactly*. Repositories with the immutable
# subject setting (the default for new repos) embed the owner and repo IDs; read the prefix with
#   gh api repos/jkatdare/SPXLcast/actions/oidc/customization/sub
# and append ":ref:refs/heads/master". Classic repos present "repo:jkatdare/SPXLcast:ref:refs/heads/master".
$prefix = (gh api repos/jkatdare/SPXLcast/actions/oidc/customization/sub | ConvertFrom-Json).sub_claim_prefix
if (-not $prefix) { $prefix = "repo:jkatdare/SPXLcast" }
@"
{ "name": "spxlcast-master",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "$($prefix):ref:refs/heads/master",
  "audiences": ["api://AzureADTokenExchange"] }
"@ | Set-Content "$env:TEMP\fc.json" -Encoding ascii
az ad app federated-credential create --id $objId --parameters "$env:TEMP\fc.json" -o none
gh secret set AZURE_CLIENT_ID --body $appId
gh secret set AZURE_TENANT_ID --body $tenant
gh secret set AZURE_SUBSCRIPTION_ID --body $sub
```

Then push the workflow file and watch it: `gh run watch`. A login failure `AADSTS700213: No matching
federated identity record found for presented assertion subject '...'` means the subject differs from
the one in the error message; create another credential with that exact subject. The workflow tags each image with the
commit hash and updates both apps to it, so a rollback is `az containerapp job update ... --image
spxlcastacr.azurecr.io/spxlcast:<older sha>`. Contributor on the resource group is the minimum
that lets it build in the registry and update the apps; it cannot touch anything outside
`spxlcast-rg`.

## 10. Health check and alerts

`.github/workflows/healthcheck.yml` checks the live site from GitHub at 15:05, 17:05, 19:05 and
22:15 UTC on weekdays (`spxlcast/health.py`, standard library only; its `CHECKS` must match the
workflow's cron lines and its `SCHEDULE` the job's, which a test enforces). It fails when:

- a scheduled job run since the check before the previous one never logged a forecast. Every run
  is judged by two checks, so one dropped or late check (GitHub delays and sometimes drops
  scheduled workflows) still reports it; Monday's first check covers Friday evening;
- `output/forecast.json` is older than the latest logged run that is at least 15 minutes old (that
  run did not finish its outputs);
- `output/live.json` is more than 10 minutes old during the session, or, outside it, stopped more
  than 10 minutes before the last close (the live loop has stopped), or is not a JSON object with
  the time it was written. The session follows the NYSE calendar in `spxlcast/config.py` that the
  loop and the page use: no ticks on holidays, 13:00 closes on early-close days;
- during the session, the quote in `live.json` has not changed for 30 minutes (its `spot_since`):
  the quote feed keeps answering with an old price. SPXL's minute quotes repeat for a few minutes
  at most;
- the latest run priced SPXL more than two sessions before today (stale price feed);
- `/healthz` is down, or `/` does not return the status page;
- the latest score step failed (the job keeps the previous track record on the page and leaves
  `output/score_error.txt`).

Data problems flagged by the latest run (the `data_flags` column, for example `fred:none`) are
reported as warnings without failing. On failure it opens an issue labelled `health-check` (later
failures comment on it) and fails the workflow run so GitHub sends its failed-workflow email. The
issue is closed by the next passing check, scheduled or manual, made while the session is open
(`health.py` tells the workflow through the step output `live_checked`): a pass at 22:15, on a
holiday, after an early close or delayed past the bell cannot see the live loop at work, so it
leaves the issue open until the next session's checks (close it by hand if the fix is certain).
Nothing to set up beyond pushing the file; the workflow's own token opens the issues. It uses about
90 of the 2,000 free Actions minutes a month.
Run it by hand with `gh workflow run healthcheck.yml`, or locally:

```powershell
py -m spxlcast.health --url https://spxlcast.com
```

## Archive

Every job run also writes, on the file share under `archive/`:

- `runs/YYYY-MM-DD/HHMMSSZ.json.gz`: the run's inputs (latest market and FRED values, holdings,
  data flags, package versions), the exact simulator arguments and the full forecast. With the same
  build, `spxlcast.archive.replay(load_run(path))` reproduces the simulation exactly. A second run
  in the same UTC second is stored as `HHMMSSZ-1.json.gz` instead of overwriting the first.
- `news/YYYY-MM.jsonl`: every headline the model scored, once per story and headline, at its first
  sighting (a headline rewritten under the same URL is stored again with the same key), with its
  text, score and relevance. Lines are ASCII JSON (non-ASCII characters are `\u` escapes; read
  them with `json.loads`). Yahoo only serves the latest headlines, so this is the only history the
  news tilt can later be calibrated on.

The status page does not serve it (it holds publishers' headline text). Download it with:

```powershell
$key = az storage account keys list -n spxlcastsa -g spxlcast-rg --query "[0].value" -o tsv
az storage file download-batch --account-name spxlcastsa --account-key $key -s spxlcast --pattern "archive/*" -d . -o none
```

It grows by roughly 100 MB a year, well within the 5 GB share.

## Updating and removing

- Code changes: push to `master` (with the workflow), or by hand the three build lines of step 2
  (they stamp a build from uncommitted changes `-dirty`) followed by
  `az containerapp job update -n spxlcast-daily -g spxlcast-rg --image spxlcastacr.azurecr.io/spxlcast:latest -o none`
  and the same `az containerapp update` for `spxlcast-web`.
- Optional, once after deploying model 0.3.0: earlier builds could archive price rows from two
  dividend bases in `.cache/archive_*.pkl` on the share. The first run of the new build rewrites
  every archived row inside the five-year download window, which is all the model reads, so this
  is not required; to rebuild the older rows too, delete those files once, with `$key` as in the
  Archive section (`az storage file delete-batch --account-name spxlcastsa --account-key $key
  -s spxlcast --pattern ".cache/archive_*.pkl"`, first with `--dryrun`). `--refresh` no longer
  resets them.
- Change the schedule: edit `CRON` in `.github/workflows/deploy.yml` and `SCHEDULE` in
  `spxlcast/health.py`, and push (or, until the next push,
  `az containerapp job update -n spxlcast-daily -g spxlcast-rg --cron-expression "..." -o none`).
- Is it alive? `https://spxlcast.com/output/live.json` should be under two minutes old during the
  session; `az containerapp logs show -n spxlcast-web -g spxlcast-rg --tail 50` shows the loop, and
  `az containerapp job execution list -n spxlcast-daily -g spxlcast-rg -o table` the hourly runs.
- Everything, including the track record on the share: `az group delete -n spxlcast-rg --yes`.
  Download `forecast_log.csv` from the status page and the archive (above) first if you want to keep them.

## Cost

The Basic registry is about $5 a month. The always-on web app (0.25 vCPU, 0.5 GiB) is the other
fixed charge: roughly $14 a month at consumption prices after the free monthly grant. The job bills
only while it runs (nine one-minute runs a day, under $1 a month), the 5 GB file share is pennies,
and GitHub Actions minutes on a private repo (deploys plus about 90 minutes a month of health
checks) are within the free monthly allowance.
