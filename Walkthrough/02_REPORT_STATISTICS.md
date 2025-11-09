# 📊 Báo Cáo Thống Kê và So Sánh OCR Engines

> Tài liệu hướng dẫn phân tích kết quả, công thức tính toán, và cách viết báo cáo khoa học

---

## 📖 Mục Lục

1. [Tổng Quan](#-tổng-quan)
2. [Dataset & Điều Kiện Thử Nghiệm](#-dataset--điều-kiện-thử-nghiệm)
3. [OCR Engines So Sánh](#-ocr-engines-so-sánh)
4. [Metrics & Công Thức](#-metrics--công-thức)
5. [Cách Đọc Kết Quả](#-cách-đọc-kết-quả)
6. [Biểu Đồ & Visualization](#-biểu-đồ--visualization)
7. [Mẫu Báo Cáo](#-mẫu-báo-cáo)

---

## 🎯 Tổng Quan

Hệ thống đánh giá độ chính xác OCR bằng cách:
1. Chạy 5 OCR engines trên cùng dataset
2. So sánh kết quả với **Ground Truth** (dữ liệu thực)
3. Tính toán các metrics chuẩn
4. Tạo biểu đồ phân tích

---

## 📁 Dataset & Điều Kiện Thử Nghiệm

### Dataset

**Tên:** Bìa Sách Tiếng Việt Dataset
- **Số lượng:** 200 ảnh bìa sách
- **Đặc điểm:**
  - Ngôn ngữ: Tiếng Việt và Tiếng Anh
  - Font chữ: Đa dạng (serif, sans-serif, handwriting)
  - Màu nền: Nhiều màu, texture phức tạp
  - Text orientation: Vertical, horizontal, tilted
  - Độ phức tạp: Cao (nhiều layers, hiệu ứng)

**Vị trí:** `OCR_Library/Bia_sach/`

**Format:** JPG, PNG (1000x1500 - 2000x3000 pixels)

### Môi Trường Thử Nghiệm

**Hardware:**
- CPU: Intel Core i5/i7 hoặc AMD Ryzen 5/7
- RAM: 8GB minimum, 16GB recommended
- GPU: Optional (CUDA-compatible cho EasyOCR, DocTR)

**Software:**
- OS: Windows 10/11, Ubuntu 20.04+, macOS 11+
- Python: 3.9+
- CUDA: 11.0+ (nếu dùng GPU)

### Ground Truth

**File:** `ground_truth.json`

**Cấu trúc:**
```json
{
  "images": [
    {
      "filename": "frieren_vol1.jpg",
      "title": "FRIEREN PHÁP SƯ TIỄN TÁNG",
      "author": "KANEHITO YAMADA",
      "publisher": "KIM ĐỒNG",
      "all_text": "BẢN ĐẶC BIỆT FRIEREN PHÁP SƯ TIỄN TÁNG VOL 1-2 NGUYÊN TÁC KANEHITO YAMADA MINH HỌA TSUKASA ABE GOU DỊCH NHÀ XUẤT BẢN KIM ĐỒNG"
    }
  ]
}
```

💡 **Quan trọng:** Field `all_text` là chuẩn để so sánh accuracy!

---

## 🔧 OCR Engines So Sánh

### 1. EasyOCR 🔵

**Đặc điểm:**
- Deep learning (CNN + RNN)
- Hỗ trợ 80+ ngôn ngữ
- GPU accelerated
- Tốt với tiếng Việt có dấu

**Thông số:**
```python
languages=['vi', 'en']
gpu=False
text_threshold=0.7
canvas_size=1280
mag_ratio=1.0
```

### 2. Tesseract OCR 🔴

**Đặc điểm:**
- LSTM-based (từ version 4.0+)
- Mã nguồn mở, phổ biến
- Tốt với text rõ ràng, nền trắng

**Thông số:**
```python
lang='vie+eng'
config='--oem 3 --psm 6'
# PSM 6 = Assume uniform text block
```

### 3. DocTR 🟢

**Đặc điểm:**
- Document Text Recognition
- Deep learning (ResNet + ViT)
- Tốt với documents có cấu trúc

**Thông số:**
```python
det_arch='db_resnet50'
reco_arch='crnn_vgg16_bn'
pretrained=True
```

### 4. GOCR 🟡

**Đặc điểm:**
- Rule-based OCR
- Nhẹ, nhanh
- Hạn chế với tiếng Việt có dấu

**Thông số:**
```python
mode=130  # Tiếng Anh + số
certainty=0.5
```

### 5. Keras OCR 🟣

**Đặc điểm:**
- Deep learning (CRAFT + CRNN)
- Tốt với scene text
- Chậm hơn các engine khác

**Thông số:**
```python
detector=keras_ocr.detection.Detector()
recognizer=keras_ocr.recognition.Recognizer()
```

---

## 📐 Metrics & Công Thức

### 1. F1-Score ⭐ (Metric Quan Trọng Nhất)

**Định nghĩa:** Trung bình điều hòa của Precision và Recall

**Công thức:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Giải thích:**
- Cân bằng giữa độ chính xác và độ phủ
- Giá trị: 0.0 - 1.0 (0% - 100%)
- 1.0 = Hoàn hảo

**Ví dụ:**
- Precision = 90%, Recall = 80%
- F1 = 2 × (0.9 × 0.8) / (0.9 + 0.8) = 0.847 = 84.7%

### 2. Precision (Độ Chính Xác)

**Định nghĩa:** Trong những từ OCR nhận ra, bao nhiêu % là ĐÚNG?

**Công thức:**
```
Precision = TP / (TP + FP)

Trong đó:
- TP (True Positive): Số từ OCR đúng
- FP (False Positive): Số từ OCR sai (nhận ra nhưng không có trong ground truth)
```

**Ví dụ:**
- OCR nhận ra: 10 từ
- Trong đó đúng: 8 từ
- Precision = 8 / 10 = 80%

### 3. Recall (Độ Phủ / Độ Nhạy)

**Định nghĩa:** Trong tất cả từ CẦN nhận ra, OCR nhận ra được bao nhiêu %?

**Công thức:**
```
Recall = TP / (TP + FN)

Trong đó:
- TP (True Positive): Số từ OCR đúng
- FN (False Negatives): Số từ bỏ sót (có trong ground truth nhưng OCR không nhận ra)
```

**Ví dụ:**
- Ground Truth: 12 từ
- OCR nhận ra đúng: 8 từ
- Recall = 8 / 12 = 66.7%

### 4. Character Accuracy (Độ Chính Xác Ký Tự)

**Định nghĩa:** So sánh độ giống nhau ở mức ký tự (dùng Levenshtein Distance)

**Công thức:**
```
Character Accuracy = 1 - (Edit Distance / Max Length)

Edit Distance = Số thao tác tối thiểu (insert, delete, substitute) 
                để biến text A thành text B
```

**Ví dụ:**
- Ground Truth: "FRIEREN"
- OCR Result: "FRIERN"
- Edit Distance: 2 (thêm 'E', xóa 'N')
- Character Accuracy = 1 - (2 / 7) = 71.4%

### 5. Processing Time (Thời Gian Xử Lý)

**Đơn vị:** Giây (seconds)

**Đo lường:**
```python
start_time = time.time()
# ... OCR processing ...
processing_time = time.time() - start_time
```

**Ý nghĩa:**
- Thời gian càng ngắn = engine càng nhanh
- Trade-off: Accuracy vs Speed

---

## 📊 Cách Đọc Kết Quả

### File JSON: `evaluation_report_*.json`

**Ví dụ kết quả:**
```json
{
  "summary": {
    "total_images": 50,
    "timestamp": "2025-01-15 10:30:00",
    "engines": {
      "easyocr": {
        "avg_f1_score": 0.8537,
        "avg_precision": 0.9124,
        "avg_recall": 0.8102,
        "avg_char_accuracy": 0.7984,
        "avg_processing_time": 4.52,
        "successful_detections": 48
      },
      "tesseract": {
        "avg_f1_score": 0.7231,
        "avg_precision": 0.7845,
        "avg_recall": 0.6812,
        "avg_char_accuracy": 0.6543,
        "avg_processing_time": 1.23,
        "successful_detections": 45
      }
    },
    "ranking": [
      {"engine": "easyocr", "f1_score": 0.8537},
      {"engine": "tesseract", "f1_score": 0.7231}
    ]
  }
}
```

### Giải Thích Kết Quả

**🥇 EasyOCR:**
- F1-Score: 85.37% → Rất tốt
- Precision: 91.24% → Rất ít false positive
- Recall: 81.02% → Phát hiện được 81% từ
- Processing Time: 4.52s → Chậm hơn Tesseract

**🥈 Tesseract:**
- F1-Score: 72.31% → Khá
- Precision: 78.45% → Nhiều false positive hơn
- Recall: 68.12% → Bỏ sót nhiều từ hơn
- Processing Time: 1.23s → Nhanh hơn EasyOCR 3.7 lần

---

## 📈 Biểu Đồ & Visualization

### 1. Detailed Bars Chart

**File:** `engine_comparison_*_detailed_bars.png`

**Nội dung:**
- So sánh 4 metrics: F1, Precision, Recall, Char Accuracy
- Grouped bar chart
- Màu sắc khác nhau cho mỗi engine

**Cách đọc:**
- Càng cao = càng tốt
- So sánh trực tiếp giữa các engines

### 2. Metrics Grid

**File:** `engine_comparison_*_metrics_grid.png`

**Nội dung:**
- 4 subplots cho 4 metrics
- Bar chart riêng cho mỗi metric

**Cách đọc:**
- Nhìn tổng quan tất cả metrics
- Dễ so sánh từng metric riêng lẻ

### 3. Speed vs Accuracy

**File:** `engine_comparison_*_speed_vs_accuracy.png`

**Nội dung:**
- Scatter plot
- Trục X: Processing Time (s)
- Trục Y: F1-Score (%)
- Bubble size: Number of detections

**Cách đọc:**
- **Góc trên bên trái:** Tốt nhất (nhanh + chính xác)
- **Góc trên bên phải:** Chính xác nhưng chậm
- **Góc dưới bên trái:** Nhanh nhưng không chính xác
- **Góc dưới bên phải:** Tệ nhất (chậm + không chính xác)

### 4. Comparison Bar Chart

**File:** `accuracy_*_comparison.png`

**Nội dung:**
- So sánh F1-Score của tất cả engines
- Sorted từ cao xuống thấp

### 5. Heatmap

**File:** `accuracy_*_heatmap.png`

**Nội dung:**
- Ma trận nhiệt cho tất cả metrics
- Màu càng đậm = giá trị càng cao

### 6. Radar Chart

**File:** `accuracy_*_radar.png`

**Nội dung:**
- Biểu đồ hình nhện
- So sánh đa chiều (4 metrics)
- Diện tích càng lớn = engine càng tốt

---

## 📝 Mẫu Báo Cáo

### Phần 1: Giới Thiệu

```
Nghiên cứu này so sánh hiệu suất của 5 OCR engines trên dataset 
gồm 200 ảnh bìa sách tiếng Việt. Các engines được đánh giá bao gồm:
EasyOCR, Tesseract, DocTR, GOCR, và Keras OCR.
```

### Phần 2: Phương Pháp

```
Dataset: 200 ảnh bìa sách tiếng Việt, độ phân giải 1000x1500 - 2000x3000 pixels.

Ground Truth: Được tạo thủ công, ghi lại toàn bộ text trên mỗi bìa sách.

Metrics: 
- F1-Score: F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- Character Accuracy: 1 - (Levenshtein Distance / Max Length)
- Processing Time: Thời gian xử lý trung bình (giây)

Điều kiện:
- Hardware: Intel i5, 8GB RAM, No GPU
- Software: Python 3.9, Windows 10
- Text normalization: Lowercase, remove punctuation, normalize whitespace
```

### Phần 3: Kết Quả

```
Bảng 1: Kết quả so sánh các OCR engines

| Engine     | F1-Score | Precision | Recall | Char Acc | Time (s) |
|------------|----------|-----------|--------|----------|----------|
| EasyOCR    | 85.37%   | 91.24%    | 81.02% | 79.84%   | 4.52     |
| Tesseract  | 72.31%   | 78.45%    | 68.12% | 65.43%   | 1.23     |
| DocTR      | 68.92%   | 74.23%    | 64.58% | 62.11%   | 3.87     |
| Keras OCR  | 45.67%   | 52.34%    | 41.23% | 55.73%   | 28.45    |
| GOCR       | 32.15%   | 41.56%    | 26.78% | 38.92%   | 0.89     |

(Xem Hình 1: engine_comparison_*_detailed_bars.png)
```

### Phần 4: Thảo Luận

```
EasyOCR đạt F1-Score cao nhất (85.37%), vượt trội so với các engines khác.
Precision của EasyOCR đạt 91.24%, cho thấy tỷ lệ false positive thấp.
Tuy nhiên, thời gian xử lý của EasyOCR (4.52s) chậm hơn Tesseract (1.23s) 
gấp 3.7 lần.

Tesseract có trade-off tốt giữa accuracy (F1=72.31%) và speed (1.23s), 
phù hợp cho ứng dụng real-time.

Keras OCR và GOCR cho kết quả kém, không phù hợp với bìa sách tiếng Việt.

Dataset gồm 200 ảnh bìa sách đa dạng về font chữ, màu sắc, và layout,
đại diện tốt cho các loại bìa sách thực tế tại Việt Nam.

(Xem Hình 2: engine_comparison_*_speed_vs_accuracy.png)
```

### Phần 5: Kết Luận

```
Nghiên cứu đã so sánh 5 OCR engines trên dataset 200 ảnh bìa sách tiếng Việt.
EasyOCR đạt accuracy cao nhất với F1-Score 85.37%, phù hợp cho ứng dụng 
yêu cầu độ chính xác cao. Tesseract là lựa chọn tốt cho ứng dụng real-time 
với F1-Score 72.31% và thời gian xử lý nhanh (1.23s).

Khuyến nghị: Sử dụng EasyOCR cho digitization projects, Tesseract cho 
real-time applications.
```

---

## 🔬 Code Tính Toán Metrics

### Precision, Recall, F1

```python
def calculate_precision_recall_f1(self, ocr_text, ground_truth_text):
    """Tính Precision, Recall, F1-Score"""
    # Normalize text
    norm_ocr = self.normalize_text(ocr_text)
    norm_gt = self.normalize_text(ground_truth_text)
    
    # Convert to word sets
    ocr_words = set(norm_ocr.split())
    gt_words = set(norm_gt.split())
    
    # True Positives: từ có trong cả OCR và ground truth
    tp = len(ocr_words.intersection(gt_words))
    
    # False Positives: từ có trong OCR nhưng không có trong GT
    fp = len(ocr_words - gt_words)
    
    # False Negatives: từ có trong GT nhưng không có trong OCR
    fn = len(gt_words - ocr_words)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1
```

### Character Accuracy

```python
def calculate_character_accuracy(self, ocr_text, ground_truth_text):
    """Tính character-level accuracy (Levenshtein distance based)"""
    norm_ocr = self.normalize_text(ocr_text)
    norm_gt = self.normalize_text(ground_truth_text)
    
    # Levenshtein distance
    distance = levenshtein_distance(norm_ocr, norm_gt)
    max_len = max(len(norm_ocr), len(norm_gt))
    
    return 1 - (distance / max_len) if max_len > 0 else 0.0

def levenshtein_distance(s1, s2):
    """Calculate edit distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]
```

---

## 📖 Tham Khảo

- EasyOCR: https://github.com/JaidedAI/EasyOCR
- Tesseract: https://github.com/tesseract-ocr/tesseract
- DocTR: https://github.com/mindee/doctr
- Keras OCR: https://github.com/faustomorales/keras-ocr
- Levenshtein Distance: https://en.wikipedia.org/wiki/Levenshtein_distance

---

**Cập nhật:** 2025-01-15  
**Tác giả:** OCR Library Team
