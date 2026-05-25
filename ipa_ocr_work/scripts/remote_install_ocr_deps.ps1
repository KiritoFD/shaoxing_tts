$ErrorActionPreference = "Continue"

$Python = "C:\Program Files\Python312\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
}
if (-not (Test-Path $Python)) {
    throw "Python 3.12 executable not found"
}

Write-Host "PYTHON=$Python"
& $Python -m pip install --upgrade pip

Write-Host "Installing TensorFlow for OCR-IPA/Calamari inference"
& $Python -m pip install "tensorflow>=2.16,<2.19"

Write-Host "Installing PaddleOCR CPU/GPU packages if available"
& $Python -m pip install paddleocr
& $Python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

Write-Host "Package availability"
$Code = @"
import importlib.util as u
for package in ["tensorflow", "paddle", "paddleocr", "transformers", "torch"]:
    print(package, bool(u.find_spec(package)))
"@
$Tmp = Join-Path $env:TEMP "check_ocr_deps_after_install.py"
Set-Content -LiteralPath $Tmp -Value $Code -Encoding UTF8
& $Python $Tmp
