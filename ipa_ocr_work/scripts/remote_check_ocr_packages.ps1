$ErrorActionPreference = "Continue"

$Code = @"
import importlib.util as u
for package in ["tensorflow", "transformers", "paddle", "paddleocr", "calamari_ocr", "torch"]:
    print(package, bool(u.find_spec(package)))
"@

$Tmp = Join-Path $env:TEMP "check_ocr_packages.py"
Set-Content -LiteralPath $Tmp -Value $Code -Encoding UTF8
py -3.12 $Tmp
