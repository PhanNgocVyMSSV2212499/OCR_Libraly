# 📚 OCR Library - Hướng Dẫn Cài Đặt và Sử Dụng

> Thư viện OCR đa nền tảng hỗ trợ 5 engines khác nhau cho việc nhận dạng văn bản tiếng Việt và tiếng Anh

---

## 📋 Giới Thiệu

**OCR_Library** là thư viện toàn diện so sánh hiệu suất của 5 OCR engines:

- 🔵 **EasyOCR** - Deep learning, hỗ trợ 80+ ngôn ngữ
- 🔴 **Tesseract** - OCR mã nguồn mở phổ biến nhất
- 🟢 **DocTR** - Document Text Recognition hiện đại
- 🟡 **GOCR** - OCR nhẹ, nhanh
- 🟣 **Keras OCR** - Deep learning với Keras

### ✨ Tính Năng Chính

- ✅ So sánh 5 OCR engines cùng lúc
- 📊 Đánh giá độ chính xác với Ground Truth
- 📈 Tạo 7 loại biểu đồ phân tích
- 🎯 Metrics: F1-Score, Precision, Recall, Character Accuracy
- ⚡ Batch processing - xử lý nhiều ảnh
- 🐋 Docker support

---

## 🚀 Cài Đặt

### Yêu Cầu Hệ Thống

- **Python:** 3.9 trở lên
- **RAM:** 4GB tối thiểu (8GB khuyến nghị)
- **Ổ cứng:** 5GB (cho models)
- **OS:** Windows 10/11, Linux, macOS

### Bước 1: Clone Repository

```bash
git clone https://github.com/DucNguyen2002-wq/OCR_Library.git
cd OCR_Library
```

### Bước 2: Tạo Virtual Environment

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt
venv\Scripts\activate        # Windows PowerShell
# hoặc
.\venv\Scripts\Activate.ps1  # Windows PowerShell

# Linux/Mac
source venv/bin/activate
```

### Bước 3: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

**Các thư viện chính:**

- `opencv-python` - Xử lý ảnh
- `easyocr` - EasyOCR engine
- `pytesseract` - Tesseract wrapper
- `doctr` - DocTR engine
- `torch`, `torchvision` - Deep learning backend
- `pandas`, `matplotlib`, `seaborn` - Phân tích & visualization

### Bước 4: Cài Tesseract OCR

#### Windows (Khuyến Nghị)

**Cách 1: Dùng WinGet**

```powershell
winget install --id UB-Mannheim.TesseractOCR
```

**Cách 2: Tải thủ công**

1. Tải từ: https://github.com/UB-Mannheim/tesseract/wiki
2. Cài đặt vào `C:\Program Files\Tesseract-OCR`
3. Script sẽ tự động tìm đường dẫn

#### Linux

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-vie
```

#### macOS

```bash
brew install tesseract tesseract-lang
```

### Bước 5: Kiểm Tra Cài Đặt

```bash
python check_setup.py
```

**Kết quả mong đợi:**

```
✅ Python version: 3.9+
✅ OpenCV: imported successfully
✅ EasyOCR: imported successfully
✅ DocTR: imported successfully
✅ Pytesseract: imported successfully
✅ Tesseract path: C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## 📖 Sử Dụng

### 1. Chạy OCR Đơn Giản

```bash
python Demo/simple_ocr.py
```

**Menu chính:**

```
========================================
        OCR COMPARISON TOOL
========================================
1. Test 1 ảnh (nhanh - có ground truth)
2. Test toàn bộ thư mục Bia_sach (200 ảnh)
3. Test thư mục tùy chỉnh
4. Thoát
========================================
```

### 2. Test Một Ảnh

Chọn **Option 1**, sau đó nhập tên ảnh:

```
Nhập tên ảnh (VD: anh_mat_trang.jpg): frieren_vol1.jpg
```

**Kết quả:**

- Hiển thị text nhận dạng của từng engine
- So sánh với Ground Truth (nếu có)
- Thời gian xử lý
- Độ chính xác (F1, Precision, Recall)

### 3. Test Toàn Bộ Dataset

Chọn **Option 2** để test tất cả ảnh trong `Bia_sach/`:

- Xử lý 200 ảnh bìa sách
- Tạo báo cáo JSON
- Tạo 7 loại biểu đồ so sánh
- Lưu kết quả vào `Results/`

### 4. Test Thư Mục Tùy Chỉnh

Chọn **Option 3**, nhập đường dẫn:

```
Nhập đường dẫn thư mục: C:\Users\...\MyImages
```

### 5. Xem Kết Quả

**Báo cáo JSON:**

- `Results/Json/ocr_results_*.json` - Kết quả OCR chi tiết
- `Results/Json/comparison_report_*.json` - Báo cáo so sánh
- `Results/evaluation_report_*.json` - Đánh giá độ chính xác

**Biểu đồ:**

- `Results/Charts/engine_comparison_*_detailed_bars.png` - So sánh chi tiết
- `Results/Charts/engine_comparison_*_metrics_grid.png` - Grid metrics
- `Results/Charts/engine_comparison_*_speed_vs_accuracy.png` - Tốc độ vs độ chính xác
- `Results/Charts/accuracy_*_comparison.png` - So sánh accuracy
- `Results/Charts/accuracy_*_heatmap.png` - Heatmap
- `Results/Charts/accuracy_*_radar.png` - Radar chart

---

## 🎯 Ground Truth & Đánh Giá

### Thêm Ground Truth Cho Ảnh Mới

```bash
python ground_truth_editor.py
```

**Chọn option 1** và nhập:

1. **Filename** - Tên file ảnh (VD: `new_book.jpg`)
2. **Title** - Tiêu đề sách
3. **Author** - Tác giả
4. **Publisher** - Nhà xuất bản
5. **All Text** - ⭐ **TOÀN BỘ TEXT** trên bìa sách (quan trọng nhất!)

💡 **Lưu ý:** Phần `all_text` cần nhập **chính xác tuyệt đối** mọi chữ trên ảnh.

### Xem/Sửa Ground Truth

```bash
python ground_truth_editor.py
```

**Các option:**

- `1` - Thêm ground truth mới
- `2` - Xem tất cả ground truth
- `3` - Tìm kiếm theo tên file
- `4` - Sửa ground truth
- `5` - Xóa ground truth

---

## 📊 Cấu Trúc Project

```
OCR_Library/
├── Demo/                          # Scripts demo
│   ├── simple_ocr.py             # Main OCR tool
│   ├── quick_ocr_test.py         # Test nhanh
│   └── json_visualization.py     # Tạo biểu đồ
├── Ocr_modules/                   # OCR engines
│   ├── easyocr_module.py         # EasyOCR
│   ├── pytesseract_module.py     # Tesseract
│   ├── doctr_module.py           # DocTR
│   ├── gocr_module.py            # GOCR
│   ├── keras_module.py           # Keras OCR
│   └── opencv_module.py          # Text detection
├── Bia_sach/                      # Dataset (200 ảnh)
├── Results/                       # Kết quả
│   ├── Charts/                   # Biểu đồ
│   └── Json/                     # Báo cáo JSON
├── Walkthrough/                   # Documentation
├── ground_truth.json             # Ground truth data
├── ocr_accuracy_evaluator.py    # Đánh giá accuracy
├── ground_truth_editor.py       # Editor tool
├── requirements.txt              # Dependencies
└── check_setup.py               # Kiểm tra cài đặt
```

---

## 🐋 Docker (Tùy Chọn)

### Build Docker Image

```bash
docker build -t ocr-library .
```

### Chạy Container

```bash
docker run -it --rm -v ${PWD}/Results:/app/Results ocr-library
```

---

## ⚙️ Tùy Chỉnh

### Thay Đổi OCR Engines

Mở `Demo/simple_ocr.py`, tìm dòng:

```python
engines_to_use = ['easyocr', 'tesseract', 'doctr']  # Chỉ chạy 3 engines
```

### Thay Đổi Ngưỡng Confidence

Mở `Ocr_modules/easyocr_module.py`:

```python
confidence_threshold=0.25  # Thay đổi ngưỡng (0-1)
```

---

## 🔧 Troubleshooting

### Lỗi: "Tesseract not found"

**Giải pháp:**

```python
# Mở Ocr_modules/pytesseract_module.py
# Thêm đường dẫn thủ công:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Lỗi: "CUDA out of memory"

**Giải pháp:**

```python
# Tắt GPU, dùng CPU
# Trong __init__ của các module:
gpu=False  # EasyOCR
device='cpu'  # DocTR
```

### EasyOCR bị đơ máy

**Giải pháp:** Đã tối ưu trong code, giảm:

- Canvas size: 2560 → 1280
- Max dimension: 1500 → 1200
- Mag ratio: 1.5 → 1.0

### Lỗi: "Module not found"

```bash
pip install -r requirements.txt --upgrade
```

---

## 📞 Liên Hệ & Hỗ Trợ

- **GitHub:** [DucNguyen2002-wq/OCR_Library](https://github.com/DucNguyen2002-wq/OCR_Library)
- **Issues:** [GitHub Issues](https://github.com/DucNguyen2002-wq/OCR_Library/issues)

---

## 📝 License

MIT License - Xem file `LICENSE` để biết thêm chi tiết.

---

## 🙏 Acknowledgments

- **EasyOCR** - JaidedAI
- **Tesseract** - Google
- **DocTR** - Mindee
- **OpenCV** - Intel
- **Keras OCR** - Keras Team
