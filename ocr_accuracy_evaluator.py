"""
OCR Accuracy Evaluator
So sánh kết quả OCR với ground truth để tính độ chính xác
"""

import json
import os
from difflib import SequenceMatcher
import re
from collections import defaultdict

class OCRAccuracyEvaluator:
    def __init__(self, ground_truth_file="ground_truth.json"):
        """Khởi tạo evaluator với file ground truth"""
        # Nếu ground_truth_file không phải là absolute path, tìm từ thư mục gốc project
        if not os.path.isabs(ground_truth_file):
            # Tìm thư mục gốc project
            base_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(base_dir, ground_truth_file)
            
            if os.path.exists(full_path):
                ground_truth_file = full_path
        
        self.ground_truth_file = ground_truth_file
        self.ground_truth_data = self.load_ground_truth()
        
    def load_ground_truth(self):
        """Đọc ground truth từ file JSON"""
        if not os.path.exists(self.ground_truth_file):
            print(f"⚠️ Không tìm thấy file ground truth: {self.ground_truth_file}")
            return {"images": []}
        
        with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Đã load {len(data['images'])} ảnh từ ground truth")
        return data
    
    def normalize_text(self, text):
        """Chuẩn hóa text để so sánh (lowercase, remove extra spaces, remove punctuation)"""
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common punctuation but keep Vietnamese characters
        text = re.sub(r'[.,;:!?\(\)\[\]{}"\'`\-–—]', '', text)
        
        # Remove numbers that might vary
        # text = re.sub(r'\d+', '', text)  # Uncomment if numbers are not important
        
        return text.strip()
    
    def calculate_character_accuracy(self, ocr_text, ground_truth_text):
        """Tính character-level accuracy (Levenshtein distance based)"""
        norm_ocr = self.normalize_text(ocr_text)
        norm_gt = self.normalize_text(ground_truth_text)
        
        if not norm_gt:
            return 0.0
        
        # Calculate edit distance
        def levenshtein_distance(s1, s2):
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
        
        distance = levenshtein_distance(norm_ocr, norm_gt)
        max_len = max(len(norm_ocr), len(norm_gt))
        
        if max_len == 0:
            return 0.0
        
        return 1 - (distance / max_len)
    
    def calculate_similarity(self, text1, text2):
        """Tính similarity giữa 2 đoạn text (0-1)"""
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def calculate_word_accuracy(self, ocr_text, ground_truth_text):
        """Tính word-level accuracy - tỷ lệ từ đúng"""
        norm_ocr = self.normalize_text(ocr_text)
        norm_gt = self.normalize_text(ground_truth_text)
        
        ocr_words = norm_ocr.split()
        gt_words = norm_gt.split()
        
        if not gt_words:
            return 0.0
        
        # Số từ đúng = số từ chung giữa OCR và ground truth
        correct_words = 0
        for gt_word in gt_words:
            if gt_word in ocr_words:
                correct_words += 1
        
        return correct_words / len(gt_words)
    
    def calculate_precision_recall_f1(self, ocr_text, ground_truth_text):
        """Tính Precision, Recall, F1-Score"""
        norm_ocr = self.normalize_text(ocr_text)
        norm_gt = self.normalize_text(ground_truth_text)
        
        ocr_words = set(norm_ocr.split())
        gt_words = set(norm_gt.split())
        
        if not ocr_words and not gt_words:
            return 1.0, 1.0, 1.0
        
        if not ocr_words:
            return 0.0, 0.0, 0.0
        
        if not gt_words:
            return 0.0, 0.0, 0.0
        
        # True Positives: từ có trong cả OCR và ground truth
        tp = len(ocr_words.intersection(gt_words))
        
        # False Positives: từ có trong OCR nhưng không có trong ground truth
        fp = len(ocr_words - gt_words)
        
        # False Negatives: từ có trong ground truth nhưng không có trong OCR
        fn = len(gt_words - ocr_words)
        
        # Precision: Trong những từ OCR nhận ra, bao nhiêu % là đúng
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall: Trong tất cả từ cần nhận ra, OCR nhận ra được bao nhiêu %
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-Score: Trung bình điều hòa của Precision và Recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def evaluate_single_image(self, filename, ocr_results):
        """
        Đánh giá kết quả OCR cho 1 ảnh
        
        Args:
            filename: Tên file ảnh
            ocr_results: Dict chứa kết quả từ các OCR engine
                {
                    'easyocr': {'text': '...', 'confidence': 0.8, ...},
                    'doctr': {...},
                    ...
                }
        
        Returns:
            Dict chứa accuracy của từng engine
        """
        # Tìm ground truth cho ảnh này
        gt_entry = None
        for img in self.ground_truth_data['images']:
            if img['filename'] == filename:
                gt_entry = img
                break
        
        if not gt_entry:
            print(f"⚠️ Không tìm thấy ground truth cho: {filename}")
            return None
        
        # Ground truth giờ là string trực tiếp
        gt_text = gt_entry['ground_truth'] if isinstance(gt_entry['ground_truth'], str) else gt_entry['ground_truth'].get('all_text', '')
        
        # Đánh giá từng engine
        results = {
            'filename': filename,
            'ground_truth': gt_text,
            'engines': {}
        }
        
        engines = ['easyocr', 'easyocr_preprocessed', 'doctr', 'doctr_preprocessed', 
                   'pytesseract', 'pytesseract_preprocessed', 'keras_ocr']
        
        for engine in engines:
            if engine not in ocr_results:
                continue
            
            engine_result = ocr_results[engine]
            
            if not engine_result.get('success'):
                results['engines'][engine] = {
                    'success': False,
                    'error': engine_result.get('error', 'Unknown error')
                }
                continue
            
            ocr_text = engine_result.get('text', '')
            
            # Tính các metric
            similarity = self.calculate_similarity(ocr_text, gt_text)
            word_accuracy = self.calculate_word_accuracy(ocr_text, gt_text)
            char_accuracy = self.calculate_character_accuracy(ocr_text, gt_text)
            precision, recall, f1 = self.calculate_precision_recall_f1(ocr_text, gt_text)
            
            results['engines'][engine] = {
                'success': True,
                'text': ocr_text,
                'similarity': similarity,
                'word_accuracy': word_accuracy,
                'char_accuracy': char_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'processing_time': engine_result.get('processing_time', 0),
                'confidence': engine_result.get('confidence', 0),
                'word_count': engine_result.get('word_count', 0)
            }
        
        return results
    
    def evaluate_batch(self, ocr_results_list):
        """
        Đánh giá batch các kết quả OCR
        
        Args:
            ocr_results_list: List các dict, mỗi dict chứa:
                {
                    'image_path': '...',
                    'image_name': '...',
                    'easyocr': {...},
                    'doctr': {...},
                    ...
                }
        
        Returns:
            Dict chứa tổng hợp accuracy của tất cả engines
        """
        all_evaluations = []
        
        for ocr_result in ocr_results_list:
            # Lấy filename
            filename = ocr_result.get('image_name', os.path.basename(ocr_result['image_path']))
            
            # Tạo dict chứa kết quả các engines (bỏ qua image_name và image_path)
            engine_results = {k: v for k, v in ocr_result.items() 
                            if k not in ['image_name', 'image_path']}
            
            evaluation = self.evaluate_single_image(filename, engine_results)
            
            if evaluation:
                all_evaluations.append(evaluation)
        
        # Tính average accuracy cho từng engine
        summary = self.calculate_summary_statistics(all_evaluations)
        
        return {
            'individual_results': all_evaluations,
            'summary': summary
        }
    
    def calculate_summary_statistics(self, evaluations):
        """Tính thống kê tổng hợp cho tất cả engines"""
        engine_stats = defaultdict(lambda: {
            'similarity_scores': [],
            'word_accuracy_scores': [],
            'char_accuracy_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'f1_scores': [],
            'processing_times': [],
            'success_count': 0,
            'total_count': 0
        })
        
        for eval_result in evaluations:
            for engine, engine_data in eval_result['engines'].items():
                stats = engine_stats[engine]
                stats['total_count'] += 1
                
                if engine_data.get('success'):
                    stats['success_count'] += 1
                    stats['similarity_scores'].append(engine_data['similarity'])
                    stats['word_accuracy_scores'].append(engine_data['word_accuracy'])
                    stats['char_accuracy_scores'].append(engine_data['char_accuracy'])
                    stats['precision_scores'].append(engine_data['precision'])
                    stats['recall_scores'].append(engine_data['recall'])
                    stats['f1_scores'].append(engine_data['f1_score'])
                    stats['processing_times'].append(engine_data['processing_time'])
        
        # Calculate averages
        summary = {}
        for engine, stats in engine_stats.items():
            if stats['similarity_scores']:
                summary[engine] = {
                    'avg_similarity': sum(stats['similarity_scores']) / len(stats['similarity_scores']),
                    'avg_word_accuracy': sum(stats['word_accuracy_scores']) / len(stats['word_accuracy_scores']),
                    'avg_char_accuracy': sum(stats['char_accuracy_scores']) / len(stats['char_accuracy_scores']),
                    'avg_precision': sum(stats['precision_scores']) / len(stats['precision_scores']),
                    'avg_recall': sum(stats['recall_scores']) / len(stats['recall_scores']),
                    'avg_f1_score': sum(stats['f1_scores']) / len(stats['f1_scores']),
                    'avg_processing_time': sum(stats['processing_times']) / len(stats['processing_times']),
                    'success_rate': stats['success_count'] / stats['total_count'],
                    'total_images': stats['total_count']
                }
            else:
                summary[engine] = {
                    'avg_similarity': 0,
                    'avg_word_accuracy': 0,
                    'avg_char_accuracy': 0,
                    'avg_precision': 0,
                    'avg_recall': 0,
                    'avg_f1_score': 0,
                    'avg_processing_time': 0,
                    'success_rate': 0,
                    'total_images': stats['total_count']
                }
        
        return summary
    
    def display_results(self, evaluation_results):
        """Hiển thị kết quả đánh giá dạng bảng"""
        print("\n" + "="*120)
        print("📊 KẾT QUẢ ĐÁNH GIÁ ĐỘ CHÍNH XÁC OCR SO VỚI GROUND TRUTH")
        print("="*120)
        
        summary = evaluation_results['summary']
        
        # Sort by F1-score (metric tổng hợp tốt nhất)
        sorted_engines = sorted(summary.items(), key=lambda x: x[1]['avg_f1_score'], reverse=True)
        
        print(f"\n{'Engine':<30} {'F1-Score':<12} {'Precision':<12} {'Recall':<12} {'Char Acc':<12} {'Time (s)':<12}")
        print("-"*120)
        
        for engine, stats in sorted_engines:
            print(f"{engine:<30} "
                  f"{stats['avg_f1_score']:<12.2%} "
                  f"{stats['avg_precision']:<12.2%} "
                  f"{stats['avg_recall']:<12.2%} "
                  f"{stats['avg_char_accuracy']:<12.2%} "
                  f"{stats['avg_processing_time']:<12.2f}")
        
        print("\n" + "="*120)
        print("📈 XẾP HẠNG ENGINES THEO F1-SCORE (Metric tổng hợp tốt nhất):")
        print("="*120)
        
        for i, (engine, stats) in enumerate(sorted_engines[:10], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}️⃣"
            print(f"{medal} {engine}:")
            print(f"   F1-Score: {stats['avg_f1_score']:.2%} | "
                  f"Precision: {stats['avg_precision']:.2%} | "
                  f"Recall: {stats['avg_recall']:.2%}")
            print(f"   Char Accuracy: {stats['avg_char_accuracy']:.2%} | "
                  f"Time: {stats['avg_processing_time']:.2f}s")
        
        print("\n" + "="*120)
        print("📝 GIẢI THÍCH METRIC:")
        print("-"*120)
        print("• F1-Score:      Điểm tổng hợp (càng cao càng tốt, 100% là hoàn hảo)")
        print("• Precision:     Trong những từ OCR nhận ra, bao nhiêu % là ĐÚNG")
        print("• Recall:        Trong tất cả từ cần nhận ra, OCR nhận ra được bao nhiêu %")
        print("• Char Accuracy: Độ chính xác ở mức ký tự (dùng Levenshtein distance)")
        print("="*120)
    
    def save_evaluation_report(self, evaluation_results, output_file="evaluation_report.json"):
        """Lưu kết quả đánh giá ra file JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Đã lưu báo cáo đánh giá: {output_file}")
    
    def create_accuracy_charts(self, evaluation_results, output_dir="Results/Charts"):
        """Tạo biểu đồ accuracy metrics"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np
        
        os.makedirs(output_dir, exist_ok=True)
        
        summary = evaluation_results['summary']
        
        # Sắp xếp theo F1-score
        sorted_engines = sorted(summary.items(), key=lambda x: x[1]['avg_f1_score'], reverse=True)
        
        engine_names = [e[0] for e in sorted_engines]
        f1_scores = [e[1]['avg_f1_score'] * 100 for e in sorted_engines]
        precisions = [e[1]['avg_precision'] * 100 for e in sorted_engines]
        recalls = [e[1]['avg_recall'] * 100 for e in sorted_engines]
        char_accs = [e[1]['avg_char_accuracy'] * 100 for e in sorted_engines]
        times = [e[1]['avg_processing_time'] for e in sorted_engines]
        
        # Tạo figure với 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('📊 ĐÁNH GIÁ ACCURACY SO VỚI GROUND TRUTH', fontsize=16, fontweight='bold')
        
        x = np.arange(len(engine_names))
        width = 0.35
        
        # 1. Biểu đồ F1-Score và Character Accuracy
        bars1 = ax1.barh(x - width/2, f1_scores, width, label='F1-Score', color='#3498db', alpha=0.8)
        bars2 = ax1.barh(x + width/2, char_accs, width, label='Char Accuracy', color='#e74c3c', alpha=0.8)
        
        ax1.set_xlabel('Độ chính xác (%)', fontweight='bold')
        ax1.set_title('🎯 F1-Score & Character Accuracy')
        ax1.set_yticks(x)
        ax1.set_yticklabels(engine_names)
        ax1.legend()
        ax1.set_xlim(0, 100)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Thêm giá trị
        for bar in bars1:
            width_bar = bar.get_width()
            ax1.text(width_bar + 1, bar.get_y() + bar.get_height()/2, 
                    f'{width_bar:.1f}%', ha='left', va='center', fontsize=8)
        
        # 2. Biểu đồ Precision và Recall
        bars3 = ax2.barh(x - width/2, precisions, width, label='Precision', color='#2ecc71', alpha=0.8)
        bars4 = ax2.barh(x + width/2, recalls, width, label='Recall', color='#f39c12', alpha=0.8)
        
        ax2.set_xlabel('Tỷ lệ (%)', fontweight='bold')
        ax2.set_title('📈 Precision & Recall')
        ax2.set_yticks(x)
        ax2.set_yticklabels(engine_names)
        ax2.legend()
        ax2.set_xlim(0, 100)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Biểu đồ Thời gian xử lý
        bars5 = ax3.barh(x, times, color='#9b59b6', alpha=0.8)
        
        ax3.set_xlabel('Thời gian (giây)', fontweight='bold')
        ax3.set_title('⏱️ Thời Gian Xử Lý')
        ax3.set_yticks(x)
        ax3.set_yticklabels(engine_names)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Thêm giá trị
        for bar in bars5:
            width_bar = bar.get_width()
            ax3.text(width_bar + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{width_bar:.2f}s', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Lưu
        output_path = os.path.join(output_dir, "accuracy_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Đã tạo biểu đồ accuracy: {output_path}")
        
        # Tạo biểu đồ radar
        self._create_accuracy_radar(sorted_engines, output_dir)
        
        return output_path
    
    def _create_accuracy_radar(self, sorted_engines, output_dir):
        """Tạo biểu đồ radar cho accuracy metrics"""
        import matplotlib.pyplot as plt
        import numpy as np
        from math import pi
        
        categories = ['F1-Score', 'Precision', 'Recall', 'Char Acc']
        N = len(categories)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        # Vẽ top 5 engines
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, (engine, data) in enumerate(sorted_engines[:5]):
            values = [
                data['avg_f1_score'],
                data['avg_precision'],
                data['avg_recall'],
                data['avg_char_accuracy']
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=engine, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('🎯 Biểu Đồ Radar - Top 5 Engines (So Với Ground Truth)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        output_path = os.path.join(output_dir, "accuracy_radar.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"🎯 Đã tạo biểu đồ radar: {output_path}")


# Test function
if __name__ == "__main__":
    evaluator = OCRAccuracyEvaluator()
    
    # Test với 1 ảnh mẫu
    test_filename = "bia_manga.jpg"
    test_ocr_results = {
        'easyocr': {
            'success': True,
            'text': 'BẢN ĐẶC BIỆT FRIEREN PHÁP SƯ TIỄN TÁNG VOL Nguyên tác: KANEHITO YAMADA Minh họa: TSUKASA ABE Gou dịch',
            'confidence': 0.818,
            'word_count': 11,
            'processing_time': 4.5
        },
        'keras_ocr': {
            'success': True,
            'text': 'ban tac biet frie wren dead ang sus 102 wol tacs nguyen kanehito yamada minh hoa tsukasa abe kmidong xuat ban nha dong gou kim dich',
            'confidence': 0.95,
            'word_count': 26,
            'processing_time': 31.6
        }
    }
    
    result = evaluator.evaluate_single_image(test_filename, test_ocr_results)
    
    if result:
        print(f"\n📊 Kết quả cho {test_filename}:")
        print(f"Ground truth: {result['ground_truth'][:100]}...")
        
        for engine, data in result['engines'].items():
            if data['success']:
                print(f"\n{engine}:")
                print(f"  F1-Score: {data['f1_score']:.2%}")
                print(f"  Precision: {data['precision']:.2%}")
                print(f"  Recall: {data['recall']:.2%}")
                print(f"  Char Accuracy: {data['char_accuracy']:.2%}")
                print(f"  Time: {data['processing_time']:.2f}s")
