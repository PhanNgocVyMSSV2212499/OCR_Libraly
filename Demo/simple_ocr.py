#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import json

# Thêm đường dẫn thư mục gốc vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Ocr_modules.easyocr_module import EasyOCRProcessor
from Ocr_modules.doctr_module import DocTRProcessor
from Ocr_modules.opencv_module import OpenCVProcessor
from Ocr_modules.pytesseract_module import PytesseractProcessor
from Ocr_modules.keras_module import KerasOCRProcessor
from Demo.simple_ocr_comparison import SimpleOCRComparisonTool
from Demo.json_visualization import JSONOCRVisualizationTool

# Import OCR Accuracy Evaluator
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from ocr_accuracy_evaluator import OCRAccuracyEvaluator

class SimpleOCRTool:
    def __init__(self):
        print("🚀 KHỞI TẠO SIMPLE OCR TOOL")
        print("="*50)
        
        # Thiết lập đường dẫn thư mục
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(self.base_dir, "Results")
        self.json_dir = os.path.join(self.results_dir, "Json")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.json_dir, exist_ok=True)
        
        # Khởi tạo các processor
        self.easyocr_processor = EasyOCRProcessor(['vi', 'en'], gpu=False)
        self.doctr_processor = DocTRProcessor(pretrained=True)
        self.opencv_processor = OpenCVProcessor()
        self.pytesseract_processor = PytesseractProcessor()
            
        # Thêm Keras OCR processor
        try:
            print("Đang khởi tạo Keras OCR...")
            self.keras_processor = KerasOCRProcessor()
            print("✓ Keras OCR đã được khởi tạo")
        except Exception as e:
            print(f"⚠️ Không thể khởi tạo Keras OCR: {e}")
            self.keras_processor = None
        
        # Khởi tạo comparison tool và visualization tool
        self.comparison_tool = SimpleOCRComparisonTool()
        self.visualization_tool = JSONOCRVisualizationTool()
        
        # Khởi tạo accuracy evaluator
        ground_truth_path = os.path.join(self.base_dir, "ground_truth.json")
        self.accuracy_evaluator = OCRAccuracyEvaluator(ground_truth_path)
        
        print("✅ Tất cả mô hình đã sẵn sàng!")
    
    def process_single_image(self, image_path):
        """Xử lý một ảnh với tất cả các phương pháp OCR"""
        image_name = os.path.basename(image_path)
        print(f"\n{'='*60}")
        print(f"🖼️  Đang xử lý: {image_name}")
        print(f"{'='*60}")
        
        # Dictionary lưu kết quả
        results = {
            'image_name': image_name,
            'image_path': image_path
        }
        
        # 1. EasyOCR (ảnh gốc)
        print("\n1️⃣ EASYOCR (ảnh gốc):")
        easyocr_result = self.easyocr_processor.extract_text(image_path, confidence_threshold=0.1)
        results['easyocr'] = easyocr_result
        
        if easyocr_result['success']:
            print(f"   ✅ Thành công")
            print(f"   ⏱️  Thời gian: {easyocr_result['processing_time']:.3f} giây")
            print(f"   📝 Số từ: {easyocr_result['word_count']}")
            print(f"   🎯 Độ chính xác: {easyocr_result['confidence']:.3f}")
            print(f"   📄 Text: {easyocr_result['text'][:100]}{'...' if len(easyocr_result['text']) > 100 else ''}")
        else:
            print(f"   ❌ Lỗi: {easyocr_result.get('error', 'Unknown error')}")
        
        # 2. EasyOCR (ảnh tiền xử lý)
        print("\n1️⃣b. EASYOCR (ảnh tiền xử lý):")
        easyocr_prep_result = self.easyocr_processor.extract_text_with_preprocessing(image_path, confidence_threshold=0.1)
        results['easyocr_preprocessed'] = easyocr_prep_result
        
        if easyocr_prep_result['success']:
            print(f"   ✅ Thành công")
            print(f"   ⏱️  Thời gian: {easyocr_prep_result['processing_time']:.3f} giây")
            print(f"   📝 Số từ: {easyocr_prep_result['word_count']}")
            print(f"   🎯 Độ chính xác: {easyocr_prep_result['confidence']:.3f}")
            print(f"   📄 Text: {easyocr_prep_result['text'][:80]}{'...' if len(easyocr_prep_result['text']) > 80 else ''}")
        else:
            print(f"   ❌ Lỗi: {easyocr_prep_result.get('error', 'Unknown error')}")
        
        # 3. DocTR (ảnh gốc)
        print("\n2️⃣ DOCTR (ảnh gốc):")
        doctr_result = self.doctr_processor.extract_text(image_path, confidence_threshold=0.1)
        results['doctr'] = doctr_result
        
        if doctr_result['success']:
            print(f"   ✅ Thành công")
            print(f"   ⏱️  Thời gian: {doctr_result['processing_time']:.3f} giây")
            print(f"   📝 Số từ: {doctr_result['word_count']}")
            print(f"   🎯 Độ chính xác: {doctr_result['confidence']:.3f}")
            print(f"   📄 Text: {doctr_result['text'][:100]}{'...' if len(doctr_result['text']) > 100 else ''}")
        else:
            print(f"   ❌ Lỗi: {doctr_result.get('error', 'Unknown error')}")
        
        # 4. DocTR (ảnh tiền xử lý)
        print("\n2️⃣b. DOCTR (ảnh tiền xử lý):")
        doctr_prep_result = self.doctr_processor.extract_text_with_preprocessing(image_path, confidence_threshold=0.1)
        results['doctr_preprocessed'] = doctr_prep_result
        
        if doctr_prep_result['success']:
            print(f"   ✅ Thành công")
            print(f"   ⏱️  Thời gian: {doctr_prep_result['processing_time']:.3f} giây")
            print(f"   📝 Số từ: {doctr_prep_result['word_count']}")
            print(f"   🎯 Độ chính xác: {doctr_prep_result['confidence']:.3f}")
            print(f"   📄 Text: {doctr_prep_result['text'][:80]}{'...' if len(doctr_prep_result['text']) > 80 else ''}")
        else:
            print(f"   ❌ Lỗi: {doctr_prep_result.get('error', 'Unknown error')}")
        
        # 5. Pytesseract (ảnh gốc)
        print("\n3️⃣ PYTESSERACT (ảnh gốc):")
        pytess_result = self.pytesseract_processor.extract_text(image_path, lang='vie+eng', confidence_threshold=30)
        results['pytesseract'] = pytess_result
        
        if pytess_result['success']:
            print(f"   ✅ Thành công")
            print(f"   ⏱️  Thời gian: {pytess_result['processing_time']:.3f} giây")
            print(f"   📝 Số từ: {pytess_result['word_count']}")
            print(f"   🎯 Độ chính xác: {pytess_result['confidence']:.3f}")
            print(f"   📄 Text: {pytess_result['text'][:100]}{'...' if len(pytess_result['text']) > 100 else ''}")
        else:
            print(f"   ❌ Lỗi: {pytess_result.get('error', 'Unknown error')}")
        
        # 6. Pytesseract (ảnh tiền xử lý - đặc biệt cho bìa sách màu)
        processed_img = self.opencv_processor.preprocess_for_ocr(image_path, 'book_cover')
        if processed_img is not None:
            print("\n3️⃣b. PYTESSERACT (ảnh tiền xử lý - tối ưu bìa sách):")
            pytess_prep_result = self.pytesseract_processor.extract_text(processed_img, lang='vie+eng', confidence_threshold=30)
            results['pytesseract_preprocessed'] = pytess_prep_result
            
            if pytess_prep_result['success']:
                print(f"   ✅ Thành công")
                print(f"   ⏱️  Thời gian: {pytess_prep_result['processing_time']:.3f} giây")
                print(f"   📝 Số từ: {pytess_prep_result['word_count']}")
                print(f"   🎯 Độ chính xác: {pytess_prep_result['confidence']:.3f}")
                print(f"   📄 Text: {pytess_prep_result['text'][:80]}{'...' if len(pytess_prep_result['text']) > 80 else ''}")
            else:
                print(f"   ❌ Lỗi: {pytess_prep_result.get('error', 'Unknown error')}")
        else:
            results['pytesseract_preprocessed'] = {'success': False, 'error': 'Preprocessing failed'}
        
        # 4. OpenCV (Text Region Detection)
        print("\n4️⃣ OPENCV (Phát hiện vùng text):")
        opencv_result = self.opencv_processor.extract_text_regions(image_path)
        results['opencv'] = opencv_result
        
        if opencv_result['success']:
            print(f"   ✅ Thành công")
            print(f"   ⏱️  Thời gian: {opencv_result['processing_time']:.3f} giây")
            print(f"   🔍 Vùng text phát hiện: {opencv_result.get('text_regions_detected', 0)}")
            print(f"   📊 Tổng contours: {opencv_result.get('total_contours', 0)}")
        else:
            print(f"   ❌ Lỗi: {opencv_result.get('error', 'Unknown error')}")
        
        # 5. Keras OCR (Không tiền xử lý - tốt nhất với ảnh gốc)
        if self.keras_processor:
            print("\n5️⃣ KERAS OCR:")
            keras_result = self.keras_processor.process_image(image_path, preprocess=False)
            results['keras_ocr'] = keras_result
            
            if keras_result['success']:
                print(f"   ✅ Thành công")
                print(f"   ⏱️  Thời gian: {keras_result['processing_time']:.3f} giây")
                print(f"   📝 Số từ: {keras_result['word_count']}")
                print(f"   🎯 Độ chính xác: {keras_result['confidence']:.3f}")
                print(f"   📄 Text: {keras_result['text'][:100]}{'...' if len(keras_result['text']) > 100 else ''}")
            else:
                print(f"   ❌ Lỗi: {keras_result.get('error', 'Unknown error')}")
        else:
            results['keras_ocr'] = {'success': False, 'error': 'Keras OCR processor not available'}
        
        return results
    
    def process_folder(self, folder_path):
        """Xử lý tất cả ảnh trong thư mục"""
        # Thử tìm thư mục từ base directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_folder_path = os.path.join(base_dir, folder_path)
        
        # Kiểm tra đường dẫn tương đối trước
        if not os.path.exists(folder_path):
            if os.path.exists(full_folder_path):
                folder_path = full_folder_path
            else:
                print(f"❌ Không tìm thấy thư mục: {folder_path}")
                print(f"❌ Cũng không tìm thấy: {full_folder_path}")
                return [], None
        
        # Tìm tất cả file ảnh
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, filename))
        
        if not image_files:
            print(f"❌ Không tìm thấy ảnh nào trong thư mục: {folder_path}")
            return [], None
        
        print(f"📁 Tìm thấy {len(image_files)} ảnh để xử lý")
        
        results = []
        total_start_time = time.time()
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]", end=" ")
            result = self.process_single_image(image_path)
            results.append(result)
        
        total_time = time.time() - total_start_time
        
        # Tạo báo cáo so sánh
        print(f"\n{'='*80}")
        print("📊 TẠO BÁO CÁO SO SÁNH")
        print(f"{'='*80}")
        
        comparison_results = self.comparison_tool.compare_ocr_results(results)
        self.comparison_tool.display_comparison_table(comparison_results)
        
        # Đánh giá độ chính xác với ground truth
        print(f"\n{'='*80}")
        print("🎯 ĐÁNH GIÁ ĐỘ CHÍNH XÁC VỚI GROUND TRUTH")
        print(f"{'='*80}")
        
        evaluation_results = self.accuracy_evaluator.evaluate_batch(results)
        self.accuracy_evaluator.display_results(evaluation_results)
        
        # Lưu báo cáo đánh giá
        eval_report_filename = f"evaluation_report_{int(time.time())}.json"
        eval_report_path = os.path.join(self.results_dir, eval_report_filename)
        self.accuracy_evaluator.save_evaluation_report(evaluation_results, eval_report_path)
        
        # Tạo biểu đồ accuracy từ evaluation results
        print(f"\n{'='*80}")
        print("📊 TẠO BIỂU ĐỒ ACCURACY")
        print(f"{'='*80}")
        try:
            chart_paths = self.visualization_tool.create_accuracy_charts_from_evaluation(
                eval_report_path, 
                output_prefix=f"accuracy_{int(time.time())}"
            )
            if chart_paths:
                print(f"\n✅ Đã tạo {len(chart_paths)} biểu đồ accuracy cơ bản:")
                for chart_type, path in chart_paths.items():
                    print(f"   - {chart_type}: {path}")
        except Exception as e:
            print(f"❌ Lỗi tạo biểu đồ accuracy: {str(e)}")
        
        # Tạo biểu đồ chi tiết so sánh từng engine
        print(f"\n{'='*80}")
        print("📊 TẠO BIỂU ĐỒ CHI TIẾT SO SÁNH ENGINES")
        print(f"{'='*80}")
        try:
            detailed_chart_paths = self.visualization_tool.create_detailed_engine_comparison(
                eval_report_path,
                output_prefix=f"engine_comparison_{int(time.time())}"
            )
            if detailed_chart_paths:
                print(f"\n✅ Đã tạo {len(detailed_chart_paths)} biểu đồ chi tiết:")
                for chart_type, path in detailed_chart_paths.items():
                    print(f"   - {chart_type}: {path}")
        except Exception as e:
            print(f"❌ Lỗi tạo biểu đồ chi tiết: {str(e)}")
        
        # Lưu báo cáo so sánh
        report_filename = f"comparison_report_{int(time.time())}.json"
        self.comparison_tool.save_comparison_report(comparison_results, report_filename)
        
        # Lưu kết quả chi tiết
        self.save_results(results, total_time)
        
        return results, report_filename
    
    def save_results(self, results, total_time):
        """Lưu kết quả ra file JSON vào thư mục Results/Json"""
        
        # Convert numpy types to native Python types
        def convert_numpy_types(obj):
            if hasattr(obj, 'dtype'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj
        
        output_data = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_images': len(results),
                'total_processing_time': total_time
            },
            'results': convert_numpy_types(results)
        }
        
        filename = f"ocr_results_{int(time.time())}.json"
        
        try:
            # Lưu vào thư mục Json
            file_path = os.path.join(self.json_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Kết quả chi tiết đã lưu: {file_path}")
        except Exception as e:
            print(f"❌ Lỗi lưu file: {str(e)}")
    
    def create_visualization(self, report_filename, output_name):
        """Tạo biểu đồ phân tích từ file JSON report"""        
        try:
            print("📊 Tạo biểu đồ cột nhóm từ JSON...")
            self.visualization_tool.create_grouped_bar_chart_from_json(report_filename, f"{output_name}_grouped_bar")
            
            print("🎯 Tạo biểu đồ radar từ JSON...")
            self.visualization_tool.create_radar_chart_from_json(report_filename, f"{output_name}_radar")
            
            print("💫 Tạo biểu đồ bong bóng từ JSON...")
            self.visualization_tool.create_bubble_chart_from_json(report_filename, f"{output_name}_bubble")
            
            print(f"✅ Tất cả biểu đồ đã được tạo với prefix: {output_name}")
            print(f"📂 Kiểm tra thư mục hiện tại để xem các file .png")
            
        except Exception as e:
            print(f"❌ Lỗi tạo biểu đồ: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    print("🚀 SIMPLE OCR TOOL")
    print("EasyOCR + DocTR + OpenCV + Pytesseract")
    print("="*40)
    
    # Menu
    print("Chọn chế độ:")
    print("1. Test một ảnh")
    print("2. Test tất cả ảnh trong thư mục Bia_sach")
    print("3. Test thư mục tùy chỉnh")
    print("0. Thoát")
    
    try:
        choice = input("\nNhập lựa chọn (0-3): ").strip()
        
        if choice == "0":
            print("👋 Tạm biệt!")
            return
        
        # Khởi tạo tool
        ocr_tool = SimpleOCRTool()
        
        if choice == "1":
            # Test một ảnh
            sample_images = [
                "bia-ngu-van-lop-12.jpg",
                "sach_tieng_anh.jpg",
                "../Bia_sach/bia_lightnovel.jpg",
                "../Bia_sach/bia_manga.jpg", 
                "../Bia_sach/Bia_sach_Harry_Potter_phan_1.jpg",
                "../Bia_sach/laptrinhweb.jpg"
            ]
            
            print("\nChọn ảnh:")
            available_images = []
            for i, img_path in enumerate(sample_images, 1):
                if os.path.exists(img_path):
                    available_images.append(img_path)
                    print(f"{i}. {os.path.basename(img_path)}")
            
            if not available_images:
                print("❌ Không tìm thấy ảnh nào!")
                return
            
            img_choice = input(f"Chọn ảnh (1-{len(available_images)}): ").strip()
            if img_choice.isdigit() and 1 <= int(img_choice) <= len(available_images):
                selected_image = available_images[int(img_choice) - 1]
                result = ocr_tool.process_single_image(selected_image)
                
                # Tạo so sánh cho một ảnh
                comparison = ocr_tool.comparison_tool.compare_ocr_results([result])
                ocr_tool.comparison_tool.display_comparison_table(comparison)
            else:
                print("❌ Lựa chọn không hợp lệ!")
        
        elif choice == "2":
            # Test thư mục Bia_sach với visualization
            bia_sach_path = os.path.join(ocr_tool.base_dir, "Bia_sach")
            results, report_filename = ocr_tool.process_folder(bia_sach_path)
            if results:
                print("\n🎨 Đang tạo biểu đồ phân tích...")
                ocr_tool.create_visualization(report_filename, "Bia_sach_analysis")
        
        elif choice == "3":
            # Test thư mục tùy chỉnh với visualization
            folder_path = input("Nhập đường dẫn thư mục: ").strip()
            results, report_filename = ocr_tool.process_folder(folder_path)
            if results:
                print("\n🎨 Đang tạo biểu đồ phân tích...")
                folder_name = os.path.basename(folder_path) or "custom_folder"
                ocr_tool.create_visualization(report_filename, f"{folder_name}_analysis")
        
        else:
            print("❌ Lựa chọn không hợp lệ!")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Đã hủy bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()