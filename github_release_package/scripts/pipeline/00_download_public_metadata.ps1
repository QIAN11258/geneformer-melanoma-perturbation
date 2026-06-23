$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$projectRoot = Split-Path -Parent $PSScriptRoot
$metadataDir = Join-Path $projectRoot "data_raw\metadata"
$logDir = Join-Path $projectRoot "logs"
New-Item -ItemType Directory -Force -Path $metadataDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$downloads = @(
    @{
        Url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115978/suppl/GSE115978_cell.annotations.csv.gz"
        File = "GSE115978_cell.annotations.csv.gz"
    },
    @{
        Url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120575/suppl/GSE120575_patient_ID_single_cells.txt.gz"
        File = "GSE120575_patient_ID_single_cells.txt.gz"
    }
)

$logPath = Join-Path $logDir "metadata_download.log"
"metadata download started: $(Get-Date -Format s)" | Set-Content -LiteralPath $logPath

foreach ($item in $downloads) {
    $outFile = Join-Path $metadataDir $item.File
    Invoke-WebRequest -Uri $item.Url -OutFile $outFile
    $size = (Get-Item -LiteralPath $outFile).Length
    "$($item.File)`t$size`t$($item.Url)" | Add-Content -LiteralPath $logPath
}

"metadata download finished: $(Get-Date -Format s)" | Add-Content -LiteralPath $logPath
