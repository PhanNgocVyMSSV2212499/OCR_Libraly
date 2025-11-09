"""
EasyOCR Module for Vietnamese Text Recognition
Optimized for book covers with enhanced parameters
"""

import easyocr
import numpy as np
import cv2
import time
from PIL import Image
from .opencv_module import OpenCVProcessor

class EasyOCRProcessor:
    def __init__(self, languages=['vi', 'en'], gpu=False):
        """
        Khởi tạo EasyOCR processor
        
        Args:
            languages: Danh sách ngôn ngữ hỗ trợ
            gpu: Sử dụng GPU hay không
        """
        print("Đang khởi tạo EasyOCR...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.opencv_processor = OpenCVProcessor()
        print("✓ EasyOCR đã được khởi tạo")
    
    def extract_text(self, image_path, confidence_threshold=0.25):
        """
        Trích xuất văn bản từ ảnh bằng EasyOCR với các tham số tối ưu
        
        Args:
            image_path: Đường dẫn đến ảnh hoặc numpy array
            confidence_threshold: Ngưỡng độ tin cậy tối thiểu (giảm xuống 0.25)
            
        Returns:
            Dictionary chứa kết quả OCR
        """
        start_time = time.time()
        
        try:
            # Tiền xử lý ảnh nếu cần
            if isinstance(image_path, str):
                # Đọc ảnh bằng PIL để xử lý tốt hơn
                img = Image.open(image_path)
                
                # Giảm kích thước ảnh để tránh đơ máy
                width, height = img.size
                max_dim = 1200  # Giảm từ 1500 xuống 1200
                
                # Chỉ resize nếu ảnh quá lớn
                if width > max_dim or height > max_dim:
                    scale = min(max_dim / width, max_dim / height)
                    new_size = (int(width * scale), int(height * scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    print(f"Đã resize ảnh từ {width}x{height} → {new_size[0]}x{new_size[1]}")
                
                # Convert sang numpy array
                img_array = np.array(img)
            else:
                img_array = image_path
            
            # Đọc text với các tham số nhẹ hơn để tránh đơ máy
            print("Đang xử lý OCR...")
            results = self.reader.readtext(
                img_array,
                # Tham số tối ưu cho text detection (giảm tải)
                text_threshold=0.7,      # Tăng lên để giảm false positive
                low_text=0.4,            # Tăng lên để giảm xử lý
                link_threshold=0.4,      # Tăng lên 
                canvas_size=1280,        # Giảm từ 2560 → 1280 (giảm tải nhiều)
                mag_ratio=1.0,           # Giảm từ 1.5 → 1.0 (không phóng to thêm)
                # Tham số cho paragraph mode
                paragraph=False,         # False để detect từng word riêng lẻ
                # Confidence threshold
                min_size=10,             # Kích thước text tối thiểu (pixels)
                contrast_ths=0.1,        # Ngưỡng tương phản
                adjust_contrast=0.5,     # Tự động điều chỉnh contrast
            )
            print(f"✓ Phát hiện được {len(results)} vùng text")
            
            # Xử lý kết quả
            text_parts = []
            confidences = []
            valid_boxes = []
            
            for (bbox, text, confidence) in results:
                if confidence >= confidence_threshold:
                    text_parts.append(text)
                    confidences.append(confidence)
                    valid_boxes.append(bbox)
            
            # Nối text với khoảng trắng
            full_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'text': full_text,
                'confidence': avg_confidence,
                'word_count': len(text_parts),
                'processing_time': processing_time,
                'engine': 'EasyOCR',
                'details': results,
                'valid_detections': len(valid_boxes)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'processing_time': processing_time,
                'engine': 'EasyOCR'
            }
    
    def extract_text_with_preprocessing(self, image_path, confidence_threshold=0.25):
        """
        Trích xuất văn bản với tiền xử lý ảnh đặc biệt cho bìa sách màu
        
        Args:
            image_path: Đường dẫn đến ảnh
            confidence_threshold: Ngưỡng độ tin cậy tối thiểu
            
        Returns:
            Dictionary chứa kết quả OCR
        """
        start_time = time.time()
        
        try:
            # Tiền xử lý ảnh với OpenCV (giữ màu)
            processed_img = self.opencv_processor.preprocess_for_ocr(image_path, 'book_cover')
            
            if processed_img is None:
                # Nếu tiền xử lý thất bại, dùng ảnh gốc
                return self.extract_text(image_path, confidence_threshold)
            
            # Chạy OCR trên ảnh đã tiền xử lý
            return self.extract_text(processed_img, confidence_threshold)
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'processing_time': processing_time,
                'engine': 'EasyOCR'
            }

