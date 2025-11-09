#!/usr/bin/env python3
"""
Keras OCR Module
Sử dụng subprocess để chạy Keras OCR trong environment riêng
"""

import time
import cv2
import numpy as np
import subprocess
import os
import json
import tempfile

class KerasOCRProcessor:
    def __init__(self):
        """
        Khởi tạo Keras OCR Processor với subprocess approach
        """
        self.model_name = "Keras OCR"
        self.is_available = False
        
        # Đường dẫn đến environment riêng và script standalone
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.keras_env_python = os.path.join(base_dir, "keras_ocr_py39", "Scripts", "python.exe")
        self.standalone_script = os.path.join(os.path.dirname(__file__), "keras_ocr_standalone.py")
        
        # Kiểm tra xem environment và script có tồn tại không
        if os.path.exists(self.keras_env_python) and os.path.exists(self.standalone_script):
            print("✓ Tìm thấy Keras OCR environment và script")
            self.is_available = True
        else:
            print(f"⚠️ Không tìm thấy Keras OCR environment tại: {self.keras_env_python}")
            print(f"⚠️ Hoặc script standalone tại: {self.standalone_script}")
        
    def process_image(self, image_path, preprocess=False):
        """
        Xử lý ảnh bằng Keras OCR qua subprocess
        
        Args:
            image_path: Đường dẫn đến ảnh
            preprocess: True để tiền xử lý ảnh, False để dùng ảnh gốc
        """
        if not self.is_available:
            return {
                'success': False,
                'error': 'Keras OCR environment không khả dụng',
                'processing_time': 0.0,
                'text': '',
                'word_count': 0,
                'confidence': 0.0,
                'engine': 'Keras OCR (Not Available)'
            }
        
        start_time = time.time()
        
        try:
            # Tạo file tạm để lưu kết quả
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                temp_output_path = temp_file.name
            
            # Chạy script standalone trong environment riêng
            cmd = [self.keras_env_python, self.standalone_script, image_path, temp_output_path]
            
            # Thêm flag preprocess nếu cần
            if preprocess:
                cmd.append('--preprocess')
            
            print(f"🔧 Chạy Keras OCR subprocess: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 phút timeout
            )
            
            processing_time = time.time() - start_time
            
            if result.returncode == 0:
                # Đọc kết quả từ file JSON
                if os.path.exists(temp_output_path):
                    with open(temp_output_path, 'r', encoding='utf-8') as f:
                        ocr_result = json.load(f)
                    
                    # Cập nhật thời gian xử lý thực tế
                    ocr_result['processing_time'] = processing_time
                    
                    # Cleanup
                    os.unlink(temp_output_path)
                    
                    return ocr_result
                else:
                    return {
                        'success': False,
                        'error': 'Không tìm thấy file kết quả',
                        'processing_time': processing_time,
                        'text': '',
                        'word_count': 0,
                        'confidence': 0.0,
                        'engine': 'Keras OCR'
                    }
            else:
                # Subprocess thất bại
                error_msg = result.stderr if result.stderr else "Unknown subprocess error"
                return {
                    'success': False,
                    'error': f"Subprocess error: {error_msg}",
                    'processing_time': processing_time,
                    'text': '',
                    'word_count': 0,
                    'confidence': 0.0,
                    'engine': 'Keras OCR'
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Keras OCR timeout (>5 phút)',
                'processing_time': time.time() - start_time,
                'text': '',
                'word_count': 0,
                'confidence': 0.0,
                'engine': 'Keras OCR'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Lỗi subprocess: {str(e)}",
                'processing_time': time.time() - start_time,
                'text': '',
                'word_count': 0,
                'confidence': 0.0,
                'engine': 'Keras OCR'
            }
        finally:
            # Cleanup file tạm nếu vẫn còn
            if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
                try:
                    os.unlink(temp_output_path)
                except:
                    pass