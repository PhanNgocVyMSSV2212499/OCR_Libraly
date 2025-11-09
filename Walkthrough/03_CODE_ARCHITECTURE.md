# 💻 Code Architecture - Giải Thích Luồng Xử Lý

> Tài liệu giải thích chi tiết cấu trúc code, luồng xử lý, và các module trong OCR Library

---

## 📖 Mục Lục

1. [Tổng Quan Kiến Trúc](#-tổng-quan-kiến-trúc)
2. [Luồng Xử Lý Chính](#-luồng-xử-lý-chính)
3. [Các Module Chi Tiết](#-các-module-chi-tiết)
4. [Flow Diagrams](#-flow-diagrams)
5. [API Reference](#-api-reference)

---

## 🏗️ Tổng Quan Kiến Trúc

### Cấu Trúc Tổng Thể

```
OCR_Library/
│
├── Demo/                           # Entry points
│   ├── simple_ocr.py              # Main application
│   ├── simple_ocr_comparison.py   # Comparison tool
│   └── json_visualization.py      # Visualization
│
├── Ocr_modules/                    # OCR Engines
│   ├── easyocr_module.py          # EasyOCR wrapper
│   ├── pytesseract_module.py      # Tesseract wrapper
│   ├── doctr_module.py            # DocTR wrapper
│   ├── gocr_module.py             # GOCR wrapper
│   ├── keras_module.py            # Keras OCR wrapper
│   └── opencv_module.py           # Image preprocessing
│
├── ocr_accuracy_evaluator.py      # Accuracy evaluation
├── ground_truth_editor.py         # Ground truth management
└── ground_truth.json              # Ground truth data
```

### Kiến Trúc 3-Layer

```
┌─────────────────────────────────────────┐
│         Presentation Layer              │
│  (simple_ocr.py, json_visualization.py) │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│          Business Logic Layer           │
│ (simple_ocr_comparison.py,              │
│  ocr_accuracy_evaluator.py)             │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│           Data Access Layer             │
│  (OCR modules, ground_truth.json)       │
└─────────────────────────────────────────┘
```

---

## 🔄 Luồng Xử Lý Chính

### 1. Khởi Động Application

**File:** `Demo/simple_ocr.py`

```python
class SimpleOCRTool:
    def __init__(self):
        # Bước 1: Thiết lập đường dẫn
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(self.base_dir, "Results")
        
        # Bước 2: Khởi tạo các OCR processors
        self.easyocr_processor = EasyOCRProcessor(['vi', 'en'], gpu=False)
        self.doctr_processor = DocTRProcessor(pretrained=True)
        self.pytesseract_processor = PytesseractProcessor()
        self.keras_processor = KerasOCRProcessor()
        
        # Bước 3: Khởi tạo tools
        self.comparison_tool = SimpleOCRComparisonTool()
        self.visualization_tool = JSONOCRVisualizationTool()
        self.accuracy_evaluator = OCRAccuracyEvaluator()
```

**Flow:**
```
Start
  ↓
Load Configuration
  ↓
Initialize EasyOCR Processor
  ↓
Initialize Tesseract Processor
  ↓
Initialize DocTR Processor
  ↓
Initialize Keras OCR Processor (Optional)
  ↓
Initialize Comparison Tool
  ↓
Initialize Visualization Tool
  ↓
Initialize Accuracy Evaluator
  ↓
Ready ✅
```

### 2. Xử Lý Một Ảnh

**Method:** `process_single_image(image_path)`

```python
def process_single_image(self, image_path):
    # Bước 1: Chuẩn bị
    image_name = os.path.basename(image_path)
    results = {'image_name': image_name, 'image_path': image_path}
    
    # Bước 2: EasyOCR (ảnh gốc)
    easyocr_result = self.easyocr_processor.extract_text(image_path)
    results['easyocr'] = easyocr_result
    
    # Bước 3: EasyOCR (ảnh tiền xử lý)
    easyocr_prep = self.easyocr_processor.extract_text_with_preprocessing(image_path)
    results['easyocr_preprocessed'] = easyocr_prep
    
    # Bước 4: Tesseract
    pytesseract_result = self.pytesseract_processor.extract_text(image_path)
    results['pytesseract'] = pytesseract_result
    
    # Bước 5: DocTR
    doctr_result = self.doctr_processor.extract_text(image_path)
    results['doctr'] = doctr_result
    
    # Bước 6: Keras OCR (nếu có)
    if self.keras_processor:
        keras_result = self.keras_processor.extract_text(image_path)
        results['keras_ocr'] = keras_result
    
    # Bước 7: Đánh giá độ chính xác
    if has_ground_truth:
        accuracy = self.accuracy_evaluator.evaluate_single_image(results)
        results['accuracy'] = accuracy
    
    return results
```

**Flow:**
```
Input: image_path
  ↓
Load Image
  ↓
┌──────────────────────┐
│   EasyOCR (Raw)      │ → Extract text → Store result
└──────────────────────┘
  ↓
┌──────────────────────┐
│ EasyOCR (Processed)  │ → Preprocess → Extract → Store
└──────────────────────┘
  ↓
┌──────────────────────┐
│   Tesseract OCR      │ → Extract text → Store result
└──────────────────────┘
  ↓
┌──────────────────────┐
│      DocTR           │ → Extract text → Store result
└──────────────────────┘
  ↓
┌──────────────────────┐
│    Keras OCR         │ → Extract text → Store result
└──────────────────────┘
  ↓
Check Ground Truth?
  ├─ Yes → Evaluate Accuracy
  └─ No  → Skip
  ↓
Return results dictionary
```

### 3. Batch Processing (Xử Lý Nhiều Ảnh)

**Method:** `process_folder(folder_path)`

```python
def process_folder(self, folder_path):
    # Bước 1: Quét thư mục
    image_files = get_all_images(folder_path)
    
    # Bước 2: Xử lý từng ảnh
    all_results = []
    for image_file in image_files:
        result = self.process_single_image(image_file)
        all_results.append(result)
    
    # Bước 3: Tạo báo cáo
    comparison_report = self.comparison_tool.create_comparison_report(all_results)
    
    # Bước 4: Lưu JSON
    save_json(comparison_report, f"comparison_report_{timestamp}.json")
    
    # Bước 5: Tạo biểu đồ
    self.visualization_tool.create_all_charts(comparison_report)
    
    # Bước 6: Đánh giá accuracy
    if has_ground_truth:
        accuracy_report = self.accuracy_evaluator.evaluate_all(all_results)
        save_json(accuracy_report, f"evaluation_report_{timestamp}.json")
```

**Flow:**
```
Input: folder_path
  ↓
Scan folder for images
  ↓
For each image:
  ├─ Process image (5 OCR engines)
  ├─ Store results
  └─ Print progress
  ↓
Create comparison report
  ├─ Calculate averages
  ├─ Rank engines
  └─ Generate statistics
  ↓
Save JSON reports
  ├─ ocr_results_*.json
  └─ comparison_report_*.json
  ↓
Generate visualizations
  ├─ Detailed bars chart
  ├─ Metrics grid
  ├─ Speed vs accuracy scatter
  ├─ Comparison bar chart
  ├─ Heatmap
  ├─ Radar chart
  └─ Table
  ↓
Evaluate accuracy (if ground truth exists)
  ├─ Calculate F1, Precision, Recall
  ├─ Calculate Character Accuracy
  ├─ Rank engines
  └─ Save evaluation_report_*.json
  ↓
Done ✅
```

---

## 🔧 Các Module Chi Tiết

### 1. EasyOCR Module

**File:** `Ocr_modules/easyocr_module.py`

**Class:** `EasyOCRProcessor`

#### Phương Thức Chính

```python
class EasyOCRProcessor:
    def __init__(self, languages=['vi', 'en'], gpu=False):
        """
        Khởi tạo EasyOCR
        
        Args:
            languages: List ngôn ngữ ['vi', 'en']
            gpu: True/False - sử dụng GPU
        """
        self.reader = easyocr.Reader(languages, gpu=gpu)
    
    def extract_text(self, image_path, confidence_threshold=0.25):
        """
        Trích xuất text từ ảnh
        
        Flow:
        1. Load ảnh (PIL Image)
        2. Resize nếu cần (max 1200px)
        3. Convert sang numpy array
        4. Chạy EasyOCR với params:
           - text_threshold=0.7
           - canvas_size=1280
           - mag_ratio=1.0
        5. Filter theo confidence_threshold
        6. Join text parts
        7. Return result dict
        
        Returns:
            {
                'success': True/False,
                'text': 'extracted text',
                'confidence': 0.85,
                'processing_time': 4.52,
                'word_count': 15,
                'engine': 'EasyOCR'
            }
        """
```

#### Tối Ưu Quan Trọng

```python
# Resize image để tránh đơ máy
width, height = img.size
max_dim = 1200  # Giảm từ 1500

if width > max_dim or height > max_dim:
    scale = min(max_dim / width, max_dim / height)
    new_size = (int(width * scale), int(height * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

# Params nhẹ hơn
results = self.reader.readtext(
    img_array,
    text_threshold=0.7,
    canvas_size=1280,  # Giảm từ 2560
    mag_ratio=1.0,     # Giảm từ 1.5
)
```

### 2. Tesseract Module

**File:** `Ocr_modules/pytesseract_module.py`

**Class:** `PytesseractProcessor`

```python
class PytesseractProcessor:
    def __init__(self):
        """
        Khởi tạo Tesseract
        
        Tự động tìm Tesseract path:
        1. Check biến môi trường TESSERACT_CMD
        2. Check các đường dẫn phổ biến:
           - C:\Program Files\Tesseract-OCR\tesseract.exe
           - D:\Tesseract\tesseract.exe
        3. Nếu không tìm thấy → Warning
        """
        self.find_tesseract_path()
    
    def extract_text(self, image_path, lang='vie+eng'):
        """
        Trích xuất text từ ảnh
        
        Flow:
        1. Load ảnh (OpenCV)
        2. Convert sang grayscale
        3. Chạy Tesseract:
           - Config: --oem 3 --psm 6
           - OEM 3: Default (LSTM)
           - PSM 6: Uniform text block
        4. Get data với image_to_data()
        5. Filter theo confidence > 0
        6. Join text
        7. Return result dict
        """
```

#### Page Segmentation Mode (PSM)

```python
# PSM Values:
# 0 = Orientation and script detection (OSD) only
# 1 = Automatic page segmentation with OSD
# 3 = Fully automatic page segmentation (default)
# 6 = Assume a single uniform block of text
# 11 = Sparse text. Find as much text as possible
# 13 = Raw line. Treat image as a single text line

config = '--oem 3 --psm 6'  # Phù hợp với bìa sách
```

### 3. DocTR Module

**File:** `Ocr_modules/doctr_module.py`

**Class:** `DocTRProcessor`

```python
class DocTRProcessor:
    def __init__(self, pretrained=True):
        """
        Khởi tạo DocTR
        
        Architecture:
        - Detection: db_resnet50
        - Recognition: crnn_vgg16_bn
        """
        self.model = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn',
            pretrained=pretrained
        )
    
    def extract_text(self, image_path):
        """
        Flow:
        1. Load ảnh (DocumentFile)
        2. Predict với model
        3. Export results
        4. Parse JSON structure
        5. Concat all words
        6. Calculate confidence
        7. Return result dict
        """
```

### 4. Accuracy Evaluator

**File:** `ocr_accuracy_evaluator.py`

**Class:** `OCRAccuracyEvaluator`

```python
class OCRAccuracyEvaluator:
    def __init__(self, ground_truth_file="ground_truth.json"):
        """Load ground truth data"""
        self.ground_truth_data = self.load_ground_truth()
    
    def evaluate_single_image(self, image_name, ocr_results):
        """
        Đánh giá độ chính xác cho 1 ảnh
        
        Flow:
        1. Tìm ground truth cho ảnh
        2. Normalize text (lowercase, remove punctuation)
        3. For each OCR engine:
           a. Calculate Precision, Recall, F1
           b. Calculate Character Accuracy
           c. Store metrics
        4. Return evaluation dict
        """
    
    def calculate_precision_recall_f1(self, ocr_text, ground_truth_text):
        """
        Tính Precision, Recall, F1-Score
        
        Algorithm:
        1. Normalize texts
        2. Split into words
        3. Convert to sets
        4. Calculate:
           - TP = intersection(ocr_words, gt_words)
           - FP = ocr_words - gt_words
           - FN = gt_words - ocr_words
        5. Precision = TP / (TP + FP)
        6. Recall = TP / (TP + FN)
        7. F1 = 2 * P * R / (P + R)
        """
    
    def calculate_character_accuracy(self, ocr_text, ground_truth_text):
        """
        Tính Character Accuracy (Levenshtein Distance)
        
        Algorithm:
        1. Normalize texts
        2. Calculate edit distance (dynamic programming)
        3. Accuracy = 1 - (distance / max_length)
        """
```

#### Levenshtein Distance Algorithm

```python
def levenshtein_distance(s1, s2):
    """
    Dynamic Programming approach
    
    Matrix:
          ""  F  R  I  E  R  E  N
      ""   0  1  2  3  4  5  6  7
      F    1  0  1  2  3  4  5  6
      R    2  1  0  1  2  3  4  5
      I    3  2  1  0  1  2  3  4
      E    4  3  2  1  0  1  2  3
      R    5  4  3  2  1  0  1  2
      N    6  5  4  3  2  1  2  1
    
    Operations:
    - Insert: +1
    - Delete: +1
    - Substitute: +1 if different, +0 if same
    """
    # Initialize first row and column
    previous_row = range(len(s2) + 1)
    
    # Fill matrix
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

### 5. Visualization Tool

**File:** `Demo/json_visualization.py`

**Class:** `JSONOCRVisualizationTool`

```python
class JSONOCRVisualizationTool:
    def create_all_charts(self, json_data):
        """
        Tạo 7 loại biểu đồ
        
        1. Detailed Bars Chart
           - 4 metrics: F1, Precision, Recall, Char Acc
           - Grouped bars
           - Color: Blue, Green, Orange, Red
        
        2. Metrics Grid
           - 2x2 subplots
           - 4 separate bar charts
        
        3. Speed vs Accuracy Scatter
           - X: Processing Time
           - Y: F1-Score
           - Bubble size: Word count
        
        4. Comparison Bar Chart
           - F1-Score comparison
           - Sorted high to low
        
        5. Heatmap
           - All metrics matrix
           - Color gradient: Green (high) → Red (low)
        
        6. Radar Chart
           - 4 metrics polygon
           - Larger area = better
        
        7. Table
           - Text table with all numbers
           - Formatted with colors
        """
```

---

## 📊 Flow Diagrams

### Main Application Flow

```
┌─────────────────────────────────────────┐
│         User starts application         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│      Initialize all OCR processors      │
│  (EasyOCR, Tesseract, DocTR, Keras)    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│          Display main menu              │
│  1. Test single image                   │
│  2. Test folder (50 images)             │
│  3. Test custom folder                  │
│  4. Exit                                │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
   Option 1             Option 2/3
        │                     │
        ▼                     ▼
┌─────────────┐    ┌──────────────────┐
│  Process 1  │    │  Process folder  │
│    image    │    │  (batch mode)    │
└──────┬──────┘    └────────┬─────────┘
       │                    │
       │                    │
       └──────┬─────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│      For each image, run 5 engines:     │
│  • EasyOCR (raw)                        │
│  • EasyOCR (preprocessed)               │
│  • Tesseract                            │
│  • DocTR                                │
│  • Keras OCR                            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│     Store results in dictionary         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   Check if ground truth exists?         │
└──────────────────┬──────────────────────┘
            ┌──────┴──────┐
            │             │
           Yes           No
            │             │
            ▼             ▼
┌───────────────────┐  ┌────────────────┐
│ Evaluate accuracy │  │  Skip eval     │
│ (F1, P, R, Char)  │  │                │
└─────────┬─────────┘  └────────┬───────┘
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│        Save JSON reports                │
│  • ocr_results_*.json                   │
│  • comparison_report_*.json             │
│  • evaluation_report_*.json             │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│       Generate 7 visualization          │
│          charts (PNG files)             │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│       Display summary & rankings        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
                 Done ✅
```

### OCR Processing Pipeline

```
Input Image
     │
     ▼
┌─────────────────┐
│  Load Image     │
│  (PIL/OpenCV)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ (Optional)
│  • Grayscale    │
│  • Denoise      │
│  • Threshold    │
│  • Resize       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Detection │
│  (Find regions) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Recognition   │
│  (Read text)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Post-processing │
│ • Filter conf   │
│ • Join words    │
│ • Clean text    │
└────────┬────────┘
         │
         ▼
    Output Text
```

---

## 📚 API Reference

### SimpleOCRTool

```python
class SimpleOCRTool:
    """Main application class"""
    
    def __init__(self):
        """Initialize all processors and tools"""
        pass
    
    def process_single_image(self, image_path: str) -> dict:
        """
        Process one image with all OCR engines
        
        Args:
            image_path: Path to image file
        
        Returns:
            dict: {
                'image_name': str,
                'easyocr': dict,
                'tesseract': dict,
                'doctr': dict,
                'keras_ocr': dict,
                'accuracy': dict (if ground truth exists)
            }
        """
        pass
    
    def process_folder(self, folder_path: str) -> list:
        """
        Process all images in folder
        
        Args:
            folder_path: Path to folder containing images
        
        Returns:
            list: List of result dicts
        """
        pass
```

### OCRAccuracyEvaluator

```python
class OCRAccuracyEvaluator:
    """Evaluate OCR accuracy against ground truth"""
    
    def evaluate_single_image(self, image_name: str, ocr_results: dict) -> dict:
        """
        Evaluate accuracy for single image
        
        Args:
            image_name: Name of image file
            ocr_results: OCR results from all engines
        
        Returns:
            dict: {
                'engine_name': {
                    'f1_score': float,
                    'precision': float,
                    'recall': float,
                    'char_accuracy': float
                }
            }
        """
        pass
    
    def evaluate_all(self, all_results: list) -> dict:
        """
        Evaluate accuracy for all images
        
        Returns:
            dict: {
                'summary': {...},
                'per_image': [...],
                'ranking': [...]
            }
        """
        pass
```

---

## 🎓 Best Practices

### 1. Error Handling

```python
try:
    result = ocr_processor.extract_text(image_path)
    if result['success']:
        # Process result
        pass
    else:
        # Handle OCR failure
        print(f"Error: {result.get('error')}")
except Exception as e:
    print(f"Exception: {e}")
    # Fallback or skip
```

### 2. Memory Management

```python
# Clear cache after processing large batches
import gc
gc.collect()

# Use context managers
with Image.open(image_path) as img:
    # Process image
    pass
# Image automatically closed
```

### 3. Path Management

```python
# Always use os.path.join for cross-platform compatibility
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "Results", "Json")

# Use absolute paths
image_path = os.path.abspath(relative_path)
```

---

## 📝 Ghi Chú Phát Triển

### Thêm OCR Engine Mới

1. Tạo file mới trong `Ocr_modules/`
2. Implement class với interface chuẩn:
   ```python
   class NewOCRProcessor:
       def __init__(self, **kwargs):
           pass
       
       def extract_text(self, image_path):
           return {
               'success': True,
               'text': '...',
               'confidence': 0.85,
               'processing_time': 2.5,
               'word_count': 10,
               'engine': 'NewOCR'
           }
   ```
3. Import trong `simple_ocr.py`
4. Thêm vào `process_single_image()` method

### Thêm Metric Mới

1. Mở `ocr_accuracy_evaluator.py`
2. Thêm method tính metric mới:
   ```python
   def calculate_new_metric(self, ocr_text, ground_truth):
       # Calculate metric
       return metric_value
   ```
3. Cập nhật `evaluate_single_image()` để gọi method mới
4. Cập nhật visualization để hiển thị metric

---

**Cập nhật:** 2025-01-15  
**Tác giả:** OCR Library Team
