#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test - Test một ảnh để kiểm tra simple_ocr hoạt động
"""

import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Demo.simple_ocr import SimpleOCRTool

def main():
    print("\n" + "="*70)
    print("🧪 QUICK TEST - SIMPLE OCR")
    print("="*70 + "\n")
    
    try:
        # Khởi tạo tool
        print("⏳ Khởi tạo SimpleOCRTool...")
        tool = SimpleOCRTool()
        print("✅ SimpleOCRTool đã khởi tạo!\n")
        
        # Tìm ảnh đầu tiên trong Bia_sach
        bia_sach_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Bia_sach"
        )
        
        image_files = [f for f in os.listdir(bia_sach_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print(f"❌ Không tìm thấy ảnh nào trong {bia_sach_dir}")
            return
        
        # Test ảnh đầu tiên
        test_image = os.path.join(bia_sach_dir, image_files[0])
        print(f"🖼️  Test ảnh: {image_files[0]}\n")
        
        # Xử lý ảnh
        result = tool.process_single_image(test_image)
        
        # Hiển thị kết quả tóm tắt
        print("\n" + "="*70)
        print("📊 KẾT QUẢ TÓNG TẮT")
        print("="*70)
        
        for engine in ['easyocr', 'doctr', 'pytesseract', 'opencv', 'keras_ocr']:
            if engine in result:
                data = result[engine]
                if data.get('success'):
                    print(f"✅ {engine.upper():20} - Thời gian: {data.get('processing_time', 0):.3f}s - Độ chính xác: {data.get('confidence', 0):.3f}")
                else:
                    print(f"❌ {engine.upper():20} - Lỗi: {data.get('error', 'Unknown')}")
        
        print("\n" + "="*70)
        print("✅ TEST HOÀN THÀNH!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ LỖI: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
