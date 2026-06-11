param(
    [string]$Image = "viape/hackaithon-c:v9-3-final",
    [string]$DataDirectory = "$PSScriptRoot\..\data",
    [string]$OutputDirectory = "$PSScriptRoot\..\output"
)

$ErrorActionPreference = "Stop"
$data = (Resolve-Path $DataDirectory).Path
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$output = (Resolve-Path $OutputDirectory).Path
$inputFile = Join-Path $data "private_test.csv"
if (-not (Test-Path $inputFile)) { $inputFile = Join-Path $data "public_test.csv" }
if (-not (Test-Path $inputFile)) { throw "Place public_test.csv or private_test.csv in $data" }
$expectedQids = @(Import-Csv $inputFile | ForEach-Object { $_.qid })

docker pull $Image
if ($LASTEXITCODE -ne 0) { throw "docker pull failed" }

for ($run = 1; $run -le 3; $run++) {
    Get-ChildItem -LiteralPath $output -Force | Remove-Item -Force -Recurse
    docker run --rm --network none -v "${data}:/data:ro" -v "${output}:/output" $Image
    if ($LASTEXITCODE -ne 0) { throw "Docker validation run $run failed" }
    $predictionFile = Join-Path $output "pred.csv"
    if (-not (Test-Path $predictionFile)) { throw "Run $run did not create pred.csv" }
    if ((Get-Content $predictionFile -TotalCount 1).Trim() -ne "qid,answer") { throw "Run $run produced an invalid header" }
    $rows = @(Import-Csv $predictionFile)
    if ($rows.Count -ne $expectedQids.Count) { throw "Run $run produced an invalid row count" }
    $actualQids = @($rows | ForEach-Object { $_.qid })
    if ((Compare-Object $expectedQids $actualQids -SyncWindow 0).Count -ne 0) { throw "Run $run did not preserve qid order" }
    if (@($rows | Where-Object { $_.answer -notmatch "^[ABCD]$" }).Count -ne 0) { throw "Run $run produced an answer outside A/B/C/D" }
    Write-Host "Run $run PASS: $($rows.Count) rows, exact qid,answer schema"
}
