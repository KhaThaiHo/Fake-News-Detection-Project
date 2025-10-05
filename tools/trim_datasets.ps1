param(
    [Parameter(Mandatory=$true)] [string]$InputPath,
    [Parameter(Mandatory=$true)] [string]$OutputPath,
    [int]$MaxMB = 100
)

$maxBytes = $MaxMB * 1MB
$reader = [System.IO.File]::OpenText($InputPath)
$fs = New-Object System.IO.FileStream($OutputPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
$writer = New-Object System.IO.StreamWriter($fs, [System.Text.Encoding]::UTF8)

try {
    $header = $reader.ReadLine()
    if ($header -ne $null) { $writer.WriteLine($header) }

    while (-not $reader.EndOfStream) {
        $line = $reader.ReadLine()
        $writer.WriteLine($line)
        $writer.Flush()
        if ($fs.Length -ge $maxBytes) { break }
    }
    Write-Host "Wrote $($fs.Length) bytes to $OutputPath"
}
finally {
    $writer.Close()
    $reader.Close()
}
