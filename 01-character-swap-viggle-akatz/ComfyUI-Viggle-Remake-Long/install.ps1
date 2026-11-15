# Installs the Viggle Remake Long nodes into a ComfyUI folder (run install.bat, or: powershell -File install.ps1 [ComfyUI folder]).
# It copies this folder into ComfyUI\custom_nodes, checks that Viggle's own Viggle Remake nodes are there too, and
# does NOT restart ComfyUI: restart it yourself afterwards.
param([string]$ComfyDir = "")
$ErrorActionPreference = "Stop"
$Pack = Split-Path -Parent $MyInvocation.MyCommand.Path
$Name = "ComfyUI-Viggle-Remake-Long"

function Find-Root([string]$dir) {
    # accept the ComfyUI folder itself, its custom_nodes folder, or a portable install's top folder
    if (-not $dir) { return $null }
    $dir = $dir.Trim().Trim('"')
    if (-not (Test-Path -LiteralPath $dir -PathType Container)) { return $null }
    $dir = (Resolve-Path -LiteralPath $dir).Path
    if ((Split-Path -Leaf $dir) -eq "custom_nodes") { $dir = Split-Path -Parent $dir }
    if (Test-Path -LiteralPath (Join-Path $dir "custom_nodes")) {
        if ((Test-Path -LiteralPath (Join-Path $dir "main.py")) -or (Test-Path -LiteralPath (Join-Path $dir "models"))) { return $dir }
    }
    if (Test-Path -LiteralPath (Join-Path $dir "ComfyUI\custom_nodes")) { return (Join-Path $dir "ComfyUI") }
    return $null
}

Write-Host ""
Write-Host "Viggle Remake Long - installer" -ForegroundColor Cyan
Write-Host "------------------------------"

$root = Find-Root $ComfyDir
if (-not $root) {
    $found = @()
    $guesses = @(
        (Join-Path (Split-Path -Parent $Pack) ".."),                       # unpacked inside ComfyUI\custom_nodes
        (Split-Path -Parent $Pack),
        (Join-Path $env:USERPROFILE "Documents\ComfyUI"),                   # ComfyUI Desktop app
        (Join-Path $env:USERPROFILE "ComfyUI"),
        (Join-Path $env:USERPROFILE "Desktop\ComfyUI_windows_portable"),
        (Join-Path $env:USERPROFILE "Downloads\ComfyUI_windows_portable")
    )
    foreach ($drive in (Get-PSDrive -PSProvider FileSystem | Select-Object -ExpandProperty Root)) {
        $guesses += (Join-Path $drive "ComfyUI"), (Join-Path $drive "ComfyUI_windows_portable")
    }
    foreach ($g in $guesses) {
        $r = Find-Root $g
        if ($r -and ($found -notcontains $r)) { $found += $r }
    }
    if ($found.Count -eq 1) {
        $root = $found[0]
        Write-Host "Found ComfyUI: $root"
        $ok = Read-Host "Install there? [Y/n]"
        if ($ok -and $ok.Trim().ToLower().StartsWith("n")) { $root = $null }
    } elseif ($found.Count -gt 1) {
        Write-Host "Found more than one ComfyUI:"
        for ($i = 0; $i -lt $found.Count; $i++) { Write-Host ("  [{0}] {1}" -f ($i + 1), $found[$i]) }
        $pick = Read-Host "Type the number to install into (or press Enter to type a folder)"
        if ($pick -match '^\d+$' -and [int]$pick -ge 1 -and [int]$pick -le $found.Count) { $root = $found[[int]$pick - 1] }
    }
    while (-not $root) {
        Write-Host ""
        $typed = Read-Host "Drag your ComfyUI folder into this window (or type its path), then press Enter"
        if (-not $typed) { Write-Host "Nothing installed."; exit 1 }
        $root = Find-Root $typed
        if (-not $root) { Write-Host "That folder has no custom_nodes folder: it doesn't look like ComfyUI. Try again." -ForegroundColor Yellow }
    }
}

$target = Join-Path $root "custom_nodes\$Name"
if ((Resolve-Path -LiteralPath $Pack).Path -eq $target) {
    Write-Host "The nodes are already in $target." -ForegroundColor Green
} else {
    if (Test-Path -LiteralPath $target) {
        Write-Host "Updating the existing copy in $target"
        Remove-Item -LiteralPath $target -Recurse -Force
    }
    New-Item -ItemType Directory -Path $target | Out-Null
    foreach ($item in @("__init__.py", "nodes.py", "align.py", "README.md", "install.bat", "install.ps1", "tests")) {
        $src = Join-Path $Pack $item
        if (Test-Path -LiteralPath $src) { Copy-Item -LiteralPath $src -Destination $target -Recurse -Force }
    }
    Write-Host "Installed into $target" -ForegroundColor Green
}

# Viggle's own nodes (RemakeConditioning etc.) are needed too
$viggle = Get-ChildItem -LiteralPath (Join-Path $root "custom_nodes") -Directory | Where-Object {
    $_.Name -ne $Name -and (Get-ChildItem -LiteralPath $_.FullName -Filter "*.py" -File -ErrorAction SilentlyContinue |
        Select-String -Pattern "RemakeConditioning" -SimpleMatch -List -ErrorAction SilentlyContinue)
} | Select-Object -First 1
$bundled = Join-Path (Split-Path -Parent $Pack) "ComfyUI-Viggle-Remake"   # shipped next to this pack in the tutorial repo
if (-not $viggle -and (Test-Path -LiteralPath (Join-Path $bundled "nodes.py"))) {
    $dest = Join-Path $root "custom_nodes\ComfyUI-Viggle-Remake"
    Copy-Item -LiteralPath $bundled -Destination $dest -Recurse -Force
    Write-Host "Installed Viggle's own Viggle Remake nodes into $dest" -ForegroundColor Green
    $viggle = Get-Item -LiteralPath $dest
} elseif ($viggle) {
    Write-Host "Viggle Remake nodes found: $($viggle.Name)" -ForegroundColor Green
}
if (-not $viggle) {
    Write-Host "Viggle's own Viggle Remake nodes are NOT installed yet." -ForegroundColor Yellow
    Write-Host "  Get them from huggingface.co/Viggle/Viggle-Remake (the comfyui folder) and put them in"
    Write-Host "  $(Join-Path $root 'custom_nodes')"
}

Write-Host ""
Write-Host "Done. Now restart ComfyUI yourself, then double-click the canvas and type: Viggle" -ForegroundColor Cyan
Write-Host ""
