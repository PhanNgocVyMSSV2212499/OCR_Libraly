<#
.SYNOPSIS
    Script tự động kích hoạt virtualenv, cài dependencies và chạy đánh giá OCR.

.DESCRIPTION
    Mặc định script tìm venv tại .\keras_ocr_env. Nó sẽ:
      - chuyển tới thư mục gốc repo
      - kích hoạt venv
      - nâng cấp pip, setuptools, wheel
      - cài requirements (nếu có)
      - chạy test_with_evaluation.py
      - (tuỳ chọn) chạy Demo\simple_ocr.py và gửi lựa chọn "2" (Test tất cả ảnh)

.PARAMETER VenvPath
    Đường dẫn tới virtualenv (mặc định .\keras_ocr_env)

.PARAMETER RunSimpleOcr
    Nếu truyền switch này, script sẽ chạy `Demo\simple_ocr.py` sau khi hoàn tất đánh giá và gửi '2' vào stdin để chọn "Test tất cả ảnh".

.EXAMPLE
    .\scripts\run_evaluation.ps1
    # Sử dụng venv mặc định, cài deps và chạy đánh giá

    .\scripts\run_evaluation.ps1 -VenvPath .\.venv -RunSimpleOcr
#>

param(
    [string]$VenvPath = '.\\keras_ocr_env',
    [switch]$RunSimpleOcr
)

try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
    $repoRoot = Resolve-Path (Join-Path $scriptDir '..')
    Set-Location $repoRoot
} catch {
    Write-Error "Không thể chuyển tới thư mục repo: $_"
    exit 1
}

Write-Host "Repo root: $(Get-Location)"
Write-Host "Sử dụng venv: $VenvPath"

$activate = Join-Path (Resolve-Path $VenvPath) 'Scripts\Activate.ps1' -ErrorAction SilentlyContinue
if (-not (Test-Path $activate)) {
    Write-Error "Không tìm thấy Activate.ps1 tại $activate. Kiểm tra đường dẫn venv hoặc tạo venv mới."
    exit 1
}

Write-Host "Kích hoạt virtualenv..."
. $activate

Write-Host "Nâng cấp pip, setuptools, wheel và cài thêm build helpers..."
python -m pip install --upgrade pip setuptools wheel setuptools_scm build | Write-Host

if (Test-Path 'requirements.txt') {
    Write-Host "Cài packages từ requirements.txt..."
    python -m pip install -r requirements.txt | Write-Host
}

if (Test-Path 'keras_requirements.txt') {
    Write-Host "Cài packages từ keras_requirements.txt..."
    python -m pip install -r keras_requirements.txt | Write-Host
}

Write-Host "Chạy đánh giá: python test_with_evaluation.py"
python test_with_evaluation.py

if ($RunSimpleOcr) {
    Write-Host "Chạy Demo\simple_ocr.py và chọn '2' (Test tất cả ảnh)..."
    # Gửi '2' vào stdin để chọn option 2 trong script interactive
    echo 2 | python .\Demo\simple_ocr.py
}

Write-Host "Hoàn tất. Kiểm tra thư mục Results/ để tìm báo cáo JSON và charts."

exit 0
