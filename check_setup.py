#!/usr/bin/env python3
"""
Script kiểm tra cài đặt OCR_Library
Chạy script này để kiểm tra xem tất cả dependencies đã được cài đặt chưa
"""

import sys
import os

def check_imports():
    """Kiểm tra các thư viện cần thiết"""
    print("🔍 KIỂM TRA CÀI ĐẶT OCR_LIBRARY")
    print("="*50)
    
    # Danh sách các thư viện cần kiểm tra
    libraries = [
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('easyocr', 'EasyOCR'),
        ('pytesseract', 'Pytesseract'),
        ('doctr', 'DocTR'),
        ('keras_ocr', 'Keras OCR'),
        ('torch', 'PyTorch'),
        ('tensorflow', 'TensorFlow'),
        ('matplotlib', 'Matplotlib')
    ]
    
    results = {}
    
    for lib, name in libraries:
        try:
            __import__(lib)
            print(f"✅ {name}: OK")
            results[lib] = True
        except ImportError as e:
            print(f"❌ {name}: THIẾU - {e}")
            results[lib] = False
        except Exception as e:
            print(f"⚠️  {name}: LỖI - {e}")
            results[lib] = False
    
    return results

def check_tesseract():
    """Kiểm tra Tesseract OCR engine"""
    print(f"\n{'='*50}")
    print("🔍 KIỂM TRA TESSERACT OCR ENGINE")
    print("="*50)
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract: OK - Version {version}")
        return True
    except Exception as e:
        print(f"❌ Tesseract: THIẾU HOẶC LỖI")
        print(f"   Error: {e}")
        print("📋 HƯỚNG DẪN CÀI ĐẶT:")
        print("   1. Tải từ: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Hoặc: choco install tesseract")
        print("   3. Hoặc: winget install --id UB-Mannheim.TesseractOCR")
        return False

def check_ocr_modules():
    """Kiểm tra các module OCR tự tạo"""
    print(f"\n{'='*50}")
    print("🔍 KIỂM TRA CÁC MODULE OCR")
    print("="*50)
    
    # Thêm thư mục gốc vào sys.path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    
    modules = [
        ('Ocr_modules.opencv_module', 'OpenCV Module'),
        ('Ocr_modules.easyocr_module', 'EasyOCR Module'),
        ('Ocr_modules.doctr_module', 'DocTR Module'), 
        ('Ocr_modules.pytesseract_module', 'Pytesseract Module'),
        ('Ocr_modules.keras_ocr_module', 'Keras OCR Module')
    ]
    
    results = {}
    
    for module, name in modules:
        try:
            __import__(module)
            print(f"✅ {name}: OK")
            results[module] = True
        except ImportError as e:
            print(f"❌ {name}: LỖI IMPORT - {e}")
            results[module] = False
        except Exception as e:
            print(f"⚠️  {name}: LỖI KHÁC - {e}")
            results[module] = False
    
    return results

def test_simple_functionality():
    """Test chức năng cơ bản"""
    print(f"\n{'='*50}")
    print("🧪 TEST CHỨC NĂNG CƠ BẢN")
    print("="*50)
    
    try:
        # Test OpenCV
        import cv2
        import numpy as np
        
        # Tạo ảnh test đơn giản
        test_image = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.putText(test_image, 'TEST', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        print("✅ Tạo ảnh test: OK")
        
        # Test EasyOCR cơ bản (không load model)
        import easyocr
        print("✅ Import EasyOCR: OK")
        
        # Test DocTR cơ bản  
        from doctr.models import ocr_predictor
        print("✅ Import DocTR: OK")
        
        print("🎉 TẤT CẢ TEST CƠ BẢN ĐỀU THÀNH CÔNG!")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi trong test: {e}")
        return False

def check_sample_images():
    """Kiểm tra ảnh mẫu"""
    print(f"\n{'='*50}")
    print("🖼️  KIỂM TRA ẢNH MẪU")
    print("="*50)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, 'Bia_sach')
    
    if not os.path.exists(images_dir):
        print("❌ Thư mục Bia_sach không tồn tại")
        return False
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ Không tìm thấy ảnh mẫu nào")
        return False
    
    print(f"✅ Tìm thấy {len(image_files)} ảnh mẫu:")
    for img in image_files[:5]:  # Hiện tối đa 5 ảnh
        print(f"   📸 {img}")
    
    if len(image_files) > 5:
        print(f"   ... và {len(image_files) - 5} ảnh khác")
    
    return True

def main():
    """Hàm chính"""
    print("🚀 KIỂM TRA SETUP OCR_LIBRARY")
    print("Phiên bản: 1.0")
    print("="*60)
    
    # Các bước kiểm tra
    checks = [
        ("Thư viện Python", check_imports),
        ("Tesseract OCR", check_tesseract),
        ("Module OCR", check_ocr_modules),
        ("Chức năng cơ bản", test_simple_functionality),
        ("Ảnh mẫu", check_sample_images)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ Lỗi khi kiểm tra {check_name}: {e}")
            results[check_name] = False
    
    # Tổng kết
    print(f"\n{'='*60}")
    print("📊 TỔNG KẾT KIỂM TRA")
    print("="*60)
    
    success_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    
    for check_name, result in results.items():
        status = "✅ THÀNH CÔNG" if result else "❌ THẤT BẠI"
        print(f"{check_name}: {status}")
    
    print(f"\n🎯 Kết quả: {success_count}/{total_count} kiểm tra thành công")
    
    if success_count == total_count:
        print("🎉 SETUP HOÀN TẤT! Bạn có thể chạy các demo trong thư mục Demo/")
        print("💡 Thử chạy: python Demo/simple_ocr.py")
    else:
        print("⚠️  CẦN KHẮC PHỤC MỘT SỐ VẤN ĐỀ TRƯỚC KHI SỬ DỤNG")
        print("📋 Xem file SETUP_GUIDE.md để biết hướng dẫn chi tiết")

if __name__ == "__main__":
    main()