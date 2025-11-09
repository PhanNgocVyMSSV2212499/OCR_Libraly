"""
Module để tạo biểu đồ phân tích OCR từ file JSON comparison report
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import pi
import os

# Thiết lập matplotlib để tránh lỗi GUI
import matplotlib
matplotlib.use('Agg')

class JSONOCRVisualizationTool:
    def __init__(self):
        """Khởi tạo JSON OCR Visualization Tool"""
        print("📊 Khởi tạo JSON OCR Visualization Tool...")
        
        # Thiết lập đường dẫn thư mục (từ Demo lên DACN)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(self.base_dir, "Results")
        self.json_dir = os.path.join(self.results_dir, "Json")
        self.charts_dir = os.path.join(self.results_dir, "Charts")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Thiết lập style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Màu sắc cho các engine
        self.engine_colors = {
            'easyocr': '#3498db',           # Blue
            'easyocr_preprocessed': '#5dade2',  # Light Blue
            'doctr': '#e74c3c',             # Red
            'doctr_preprocessed': '#ec7063',    # Light Red
            'pytesseract': '#f39c12',       # Orange
            'pytesseract_preprocessed': '#f7dc6f',  # Light Orange
            'opencv': '#27ae60',            # Green
            'gocr': '#9b59b6',             # Purple
            'keras_ocr': '#111'         # Dark Orange
        }
        
        print(f"📂 Thư mục JSON: {self.json_dir}")
        print(f"📊 Thư mục Charts: {self.charts_dir}")
        print("✅ JSON OCR Visualization Tool đã sẵn sàng!")
    
    def load_comparison_data(self, json_file_path):
        """Đọc dữ liệu từ file JSON comparison report"""
        try:
            # Nếu chỉ là tên file, tìm trong thư mục Json
            if not os.path.isabs(json_file_path):
                json_file_path = os.path.join(self.json_dir, json_file_path)
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ Đã đọc dữ liệu từ: {json_file_path}")
            return data
        except Exception as e:
            print(f"❌ Lỗi đọc file JSON: {str(e)}")
            return None
    
    def filter_engines_with_data(self, data, require_confidence=True, require_words=True):
        """Lọc các engine có đủ dữ liệu cần thiết"""
        filtered_engines = {}
        
        for engine_key, engine_data in data.get('engines', {}).items():
            # Kiểm tra dữ liệu cơ bản
            has_time = engine_data.get('avg_processing_time', 0) > 0 or engine_data.get('avg_time', 0) > 0
            has_words = engine_data.get('avg_word_count', 0) > 0 or engine_data.get('avg_words', 0) > 0
            has_confidence = 'avg_confidence' in engine_data and engine_data.get('avg_confidence', 0) > 0
            
            # OpenCV chỉ có time, không có confidence và word count thực sự
            is_opencv = 'opencv' in engine_key.lower()
            
            # Điều kiện lọc
            if not has_time:
                continue
                
            if require_confidence and not has_confidence and not is_opencv:
                continue
                
            if require_words and not has_words and not is_opencv:
                continue
            
            # Chuẩn hóa tên trường để tương thích
            normalized_data = dict(engine_data)
            if 'avg_time' in engine_data and 'avg_processing_time' not in engine_data:
                normalized_data['avg_processing_time'] = engine_data['avg_time']
            if 'avg_words' in engine_data and 'avg_word_count' not in engine_data:
                normalized_data['avg_word_count'] = engine_data['avg_words']
                
            filtered_engines[engine_key] = normalized_data
        
        return filtered_engines
    
    def create_grouped_bar_chart_from_json(self, json_file_path, output_filename="grouped_bar_chart"):
        """Tạo biểu đồ cột nhóm từ file JSON"""
        print(f"📊 Tạo biểu đồ cột nhóm từ {json_file_path}...")
        
        # Đọc dữ liệu
        data = self.load_comparison_data(json_file_path)
        if not data:
            return
        
        # Lọc engines có dữ liệu - chỉ yêu cầu time, không bắt buộc confidence/words cho tất cả
        filtered_engines = {}
        for engine_key, engine_data in data.get('engines', {}).items():
            has_time = engine_data.get('avg_processing_time', 0) > 0 or engine_data.get('avg_time', 0) > 0
            if has_time:
                # Chuẩn hóa tên trường
                normalized_data = dict(engine_data)
                if 'avg_time' in engine_data and 'avg_processing_time' not in engine_data:
                    normalized_data['avg_processing_time'] = engine_data['avg_time']
                if 'avg_words' in engine_data and 'avg_word_count' not in engine_data:
                    normalized_data['avg_word_count'] = engine_data['avg_words']
                filtered_engines[engine_key] = normalized_data
        
        if not filtered_engines:
            print("❌ Không có engine nào có dữ liệu thời gian để vẽ biểu đồ cột nhóm")
            return
        
        print(f"✅ Tìm thấy {len(filtered_engines)} engines: {list(filtered_engines.keys())}")
        
        # Chuẩn bị dữ liệu và sắp xếp theo confidence từ cao đến thấp
        engine_data_list = []
        
        for engine_key, engine_data in filtered_engines.items():
            engine_info = {
                'key': engine_key,
                'name': engine_data.get('name', engine_key),
                'time': engine_data.get('avg_processing_time', 0),
                'word_count': engine_data.get('avg_word_count', 0),
                'confidence': engine_data.get('avg_confidence', 0),
                'successful': engine_data.get('successful_runs', 0),
                'failed': engine_data.get('failed_runs', 0)
            }
            
            # Tính success rate
            total = engine_info['successful'] + engine_info['failed']
            engine_info['success_rate'] = (engine_info['successful'] / total * 100) if total > 0 else 0
            
            engine_data_list.append(engine_info)
        
        # Sắp xếp theo confidence từ cao đến thấp
        engine_data_list.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Tạo các danh sách đã sắp xếp
        engine_keys = [item['key'] for item in engine_data_list]
        engine_names = [item['name'] for item in engine_data_list]
        times = [item['time'] for item in engine_data_list]
        word_counts = [item['word_count'] for item in engine_data_list]
        confidences = [item['confidence'] for item in engine_data_list]
        success_rates = [item['success_rate'] for item in engine_data_list]
        
        # Tạo subplot với 2x2 grid
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 Phân Tích Hiệu Suất OCR - So Sánh Các Thư Viện', fontsize=16, fontweight='bold')
        
        x = np.arange(len(engine_names))
        colors = [self.engine_colors.get(key, '#888888') for key in engine_keys]
        
        # 1. Biểu đồ thời gian xử lý
        bars1 = ax1.bar(x, times, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Thư viện OCR', fontweight='bold')
        ax1.set_ylabel('Thời gian (giây)', fontweight='bold')
        ax1.set_title('⏱️ Thời Gian Xử Lý Trung Bình')
        ax1.set_xticks(x)
        ax1.set_xticklabels(engine_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Thêm giá trị trên cột
        for bar, time_val in zip(bars1, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # 2. Biểu đồ số từ nhận diện (chỉ engines có word count > 0)
        valid_word_indices = [i for i, wc in enumerate(word_counts) if wc > 0]
        if valid_word_indices:
            valid_names = [engine_names[i] for i in valid_word_indices]
            valid_word_counts = [word_counts[i] for i in valid_word_indices]
            valid_colors = [colors[i] for i in valid_word_indices]
            
            x_words = np.arange(len(valid_names))
            bars2 = ax2.bar(x_words, valid_word_counts, color=valid_colors, alpha=0.8, edgecolor='black')
            ax2.set_xlabel('Thư viện OCR', fontweight='bold')
            ax2.set_ylabel('Số từ trung bình', fontweight='bold')
            ax2.set_title('📝 Số Từ Nhận Diện Trung Bình')
            ax2.set_xticks(x_words)
            ax2.set_xticklabels(valid_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Thêm giá trị trên cột
            for bar, word_val in zip(bars2, valid_word_counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{word_val:.1f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Không có dữ liệu\nsố từ', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('📝 Số Từ Nhận Diện Trung Bình')
        
        # 3. Biểu đồ độ chính xác (chỉ engines có confidence > 0)
        valid_conf_indices = [i for i, conf in enumerate(confidences) if conf > 0]
        if valid_conf_indices:
            valid_conf_names = [engine_names[i] for i in valid_conf_indices]
            valid_confidences = [confidences[i] for i in valid_conf_indices]
            valid_conf_colors = [colors[i] for i in valid_conf_indices]
            
            x_conf = np.arange(len(valid_conf_names))
            bars3 = ax3.bar(x_conf, valid_confidences, color=valid_conf_colors, alpha=0.8, edgecolor='black')
            ax3.set_xlabel('Thư viện OCR', fontweight='bold')
            ax3.set_ylabel('Độ chính xác (0-1)', fontweight='bold')
            ax3.set_title('🎯 Độ Chính Xác Trung Bình')
            ax3.set_xticks(x_conf)
            ax3.set_xticklabels(valid_conf_names, rotation=45, ha='right')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            
            # Thêm giá trị trên cột
            for bar, conf_val in zip(bars3, valid_confidences):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{conf_val:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Không có dữ liệu\nđộ chính xác', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('🎯 Độ Chính Xác Trung Bình')
        
        # 4. Biểu đồ tỷ lệ thành công
        bars4 = ax4.bar(x, success_rates, color=colors, alpha=0.8, edgecolor='black')
        ax4.set_xlabel('Thư viện OCR', fontweight='bold')
        ax4.set_ylabel('Tỷ lệ thành công (%)', fontweight='bold')
        ax4.set_title('✅ Tỷ Lệ Thành Công')
        ax4.set_xticks(x)
        ax4.set_xticklabels(engine_names, rotation=45, ha='right')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        # Thêm giá trị trên cột
        for bar, success_val in zip(bars4, success_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{success_val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Lưu biểu đồ vào thư mục Charts
        output_path = os.path.join(self.charts_dir, f"{output_filename}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Biểu đồ cột nhóm đã lưu: {output_path}")
        return output_path
    
    def create_radar_chart_from_json(self, json_file_path, output_filename="radar_chart"):
        """Tạo biểu đồ radar từ file JSON"""
        print(f"🎯 Tạo biểu đồ radar từ {json_file_path}...")
        
        # Đọc dữ liệu
        data = self.load_comparison_data(json_file_path)
        if not data:
            return
        
        # Lọc engines có dữ liệu
        filtered_engines = self.filter_engines_with_data(data, require_confidence=True, require_words=True)
        
        if not filtered_engines:
            print("❌ Không có engine nào có đủ dữ liệu để vẽ biểu đồ radar")
            return
        
        # Sắp xếp engines theo confidence từ cao đến thấp
        engine_items = list(filtered_engines.items())
        engine_items.sort(key=lambda x: x[1].get('avg_confidence', 0), reverse=True)
        sorted_engines = dict(engine_items)
        
        # Chuẩn bị dữ liệu
        categories = ['Tốc độ', 'Số từ', 'Độ chính xác', 'Tỷ lệ thành công']
        N = len(categories)
        
        # Tìm giá trị max để normalize
        max_time = max([engine_data.get('avg_processing_time', 0) for engine_data in sorted_engines.values()])
        max_words = max([engine_data.get('avg_word_count', 0) for engine_data in sorted_engines.values()])
        max_confidence = max([engine_data.get('avg_confidence', 0) for engine_data in sorted_engines.values()])
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        fig.suptitle('🎯 Biểu Đồ Radar - So Sánh Toàn Diện Các Thư Viện OCR', fontsize=16, fontweight='bold')
        
        # Góc cho mỗi trục
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        # Vẽ cho từng engine (đã sắp xếp theo confidence)
        for engine_key, engine_data in sorted_engines.items():
            # Tính toán điểm số (0-1 scale)
            if max_time > 0:
                speed_score = 1 - (engine_data.get('avg_processing_time', 0) / max_time)  # Đảo ngược: nhanh hơn = cao hơn
            else:
                speed_score = 1.0
                
            if max_words > 0:
                word_score = engine_data.get('avg_word_count', 0) / max_words
            else:
                word_score = 0.0
                
            if max_confidence > 0:
                confidence_score = engine_data.get('avg_confidence', 0) / max_confidence
            else:
                confidence_score = 0.0
            
            # Tính success rate
            successful = engine_data.get('successful_runs', 0)
            total = successful + engine_data.get('failed_runs', 0)
            success_score = (successful / total) if total > 0 else 0
            
            values = [speed_score, word_score, confidence_score, success_score]
            values += values[:1]  # Đóng vòng tròn
            
            color = self.engine_colors.get(engine_key, '#888888')
            label = engine_data.get('name', engine_key)
            
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Thiết lập trục
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        
        # Thiết lập grid
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
        ax.grid(True)
        
        # Legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        
        plt.tight_layout()
        
        # Lưu biểu đồ vào thư mục Charts
        output_path = os.path.join(self.charts_dir, f"{output_filename}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Biểu đồ radar đã lưu: {output_path}")
        return output_path
    
    def create_bubble_chart_from_json(self, json_file_path, output_filename="bubble_chart"):
        """Tạo biểu đồ bong bóng từ file JSON"""
        print(f"💫 Tạo biểu đồ bong bóng từ {json_file_path}...")
        
        # Đọc dữ liệu
        data = self.load_comparison_data(json_file_path)
        if not data:
            return
        
        # Lọc engines có dữ liệu (cần cả confidence và words)
        filtered_engines = self.filter_engines_with_data(data, require_confidence=True, require_words=True)
        
        if not filtered_engines:
            print("❌ Không có engine nào có đủ dữ liệu để vẽ biểu đồ bong bóng")
            return
        
        # Sắp xếp engines theo confidence từ cao đến thấp
        engine_items = list(filtered_engines.items())
        engine_items.sort(key=lambda x: x[1].get('avg_confidence', 0), reverse=True)
        sorted_engines = dict(engine_items)
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('💫 Biểu Đồ Bong Bóng - Tốc Độ vs Độ Chính Xác\n(Kích thước bong bóng = Số từ nhận diện)', 
                     fontsize=14, fontweight='bold')
        
        # Chuẩn bị dữ liệu (đã sắp xếp theo confidence)
        x_values = []  # Thời gian xử lý
        y_values = []  # Độ chính xác
        sizes = []     # Số từ (kích thước bong bóng)
        colors = []
        labels = []
        
        for engine_key, engine_data in sorted_engines.items():
            x_values.append(engine_data.get('avg_processing_time', 0))
            y_values.append(engine_data.get('avg_confidence', 0))
            sizes.append(engine_data.get('avg_word_count', 0) * 20)  # Scale cho bubble size
            colors.append(self.engine_colors.get(engine_key, '#888888'))
            labels.append(engine_data.get('name', engine_key))
        
        # Vẽ bubbles
        scatter = ax.scatter(x_values, y_values, s=sizes, c=colors, alpha=0.6, edgecolors='black')
        
        # Thêm labels cho từng bubble
        for i, label in enumerate(labels):
            ax.annotate(label, (x_values[i], y_values[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Thiết lập trục
        ax.set_xlabel('⏱️ Thời gian xử lý (giây)', fontsize=12, fontweight='bold')
        ax.set_ylabel('🎯 Độ chính xác', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Thêm chú thích về kích thước bubble
        word_counts = [engine_data.get('avg_word_count', 0) for engine_data in sorted_engines.values()]
        if word_counts:
            max_words = max(word_counts)
            min_words = min([w for w in word_counts if w > 0]) if any(w > 0 for w in word_counts) else 1
            
            # Legend cho bubble size
            legend_sizes = [min_words, max_words]
            legend_bubbles = []
            for size in legend_sizes:
                legend_bubbles.append(plt.scatter([], [], s=size*20, c='gray', alpha=0.6))
            
            legend1 = ax.legend(legend_bubbles, [f'{int(min_words)} từ', f'{int(max_words)} từ'], 
                               title="Kích thước bong bóng", loc='upper right', 
                               bbox_to_anchor=(1, 1))
            ax.add_artist(legend1)
        
        plt.tight_layout()
        
        # Lưu biểu đồ vào thư mục Charts
        output_path = os.path.join(self.charts_dir, f"{output_filename}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Biểu đồ bong bóng đã lưu: {output_path}")
        return output_path
    
    def create_accuracy_charts_from_evaluation(self, evaluation_json_path, output_prefix="accuracy"):
        """
        Tạo biểu đồ đánh giá accuracy từ file evaluation_report JSON
        
        Args:
            evaluation_json_path: Đường dẫn đến file evaluation_report_*.json
            output_prefix: Tiền tố cho tên file output
        """
        print(f"📊 Tạo biểu đồ accuracy từ {evaluation_json_path}...")
        
        # Đọc dữ liệu evaluation
        try:
            if not os.path.isabs(evaluation_json_path):
                # Thử tìm trong thư mục Results
                eval_path = os.path.join(self.results_dir, evaluation_json_path)
                if not os.path.exists(eval_path):
                    eval_path = evaluation_json_path
            else:
                eval_path = evaluation_json_path
            
            with open(eval_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ Đã đọc dữ liệu evaluation")
        except Exception as e:
            print(f"❌ Lỗi đọc file: {str(e)}")
            return None
        
        summary = data.get('summary', {})
        if not summary:
            print("❌ Không có dữ liệu summary trong file evaluation")
            return None
        
        # Sắp xếp theo F1-score
        sorted_engines = sorted(summary.items(), key=lambda x: x[1].get('avg_f1_score', 0), reverse=True)
        
        engine_names = [e[0].replace('_', ' ').title() for e in sorted_engines]
        f1_scores = [e[1].get('avg_f1_score', 0) * 100 for e in sorted_engines]
        precisions = [e[1].get('avg_precision', 0) * 100 for e in sorted_engines]
        recalls = [e[1].get('avg_recall', 0) * 100 for e in sorted_engines]
        char_accs = [e[1].get('avg_char_accuracy', 0) * 100 for e in sorted_engines]
        times = [e[1].get('avg_processing_time', 0) for e in sorted_engines]
        
        # === 1. Biểu đồ Bar Chart - So sánh metrics ===
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('📊 ĐÁNH GIÁ ACCURACY SO VỚI GROUND TRUTH', fontsize=16, fontweight='bold')
        
        x = np.arange(len(engine_names))
        width = 0.35
        
        # Chart 1: F1-Score và Character Accuracy
        bars1 = ax1.barh(x - width/2, f1_scores, width, label='F1-Score', color='#3498db', alpha=0.8)
        bars2 = ax1.barh(x + width/2, char_accs, width, label='Char Accuracy', color='#e74c3c', alpha=0.8)
        
        ax1.set_xlabel('Độ chính xác (%)', fontweight='bold')
        ax1.set_title('🎯 F1-Score & Character Accuracy')
        ax1.set_yticks(x)
        ax1.set_yticklabels(engine_names, fontsize=9)
        ax1.legend()
        ax1.set_xlim(0, 100)
        ax1.grid(True, alpha=0.3, axis='x')
        
        for bar in bars1:
            width_bar = bar.get_width()
            if width_bar > 0:
                ax1.text(width_bar + 1, bar.get_y() + bar.get_height()/2, 
                        f'{width_bar:.1f}%', ha='left', va='center', fontsize=8)
        
        # Chart 2: Precision và Recall
        bars3 = ax2.barh(x - width/2, precisions, width, label='Precision', color='#2ecc71', alpha=0.8)
        bars4 = ax2.barh(x + width/2, recalls, width, label='Recall', color='#f39c12', alpha=0.8)
        
        ax2.set_xlabel('Tỷ lệ (%)', fontweight='bold')
        ax2.set_title('📈 Precision & Recall')
        ax2.set_yticks(x)
        ax2.set_yticklabels(engine_names, fontsize=9)
        ax2.legend()
        ax2.set_xlim(0, 100)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Chart 3: Thời gian xử lý
        bars5 = ax3.barh(x, times, color='#9b59b6', alpha=0.8)
        
        ax3.set_xlabel('Thời gian (giây)', fontweight='bold')
        ax3.set_title('⏱️ Thời Gian Xử Lý')
        ax3.set_yticks(x)
        ax3.set_yticklabels(engine_names, fontsize=9)
        ax3.grid(True, alpha=0.3, axis='x')
        
        for bar in bars5:
            width_bar = bar.get_width()
            if width_bar > 0:
                ax3.text(width_bar + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{width_bar:.2f}s', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        output_path1 = os.path.join(self.charts_dir, f"{output_prefix}_comparison.png")
        plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Biểu đồ so sánh: {output_path1}")
        
        # === 2. Biểu đồ Radar - Top 5 engines ===
        categories = ['F1-Score', 'Precision', 'Recall', 'Char Acc']
        N = len(categories)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        colors_radar = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, (engine_key, engine_data) in enumerate(sorted_engines[:5]):
            values = [
                engine_data.get('avg_f1_score', 0),
                engine_data.get('avg_precision', 0),
                engine_data.get('avg_recall', 0),
                engine_data.get('avg_char_accuracy', 0)
            ]
            values += values[:1]
            
            engine_label = engine_key.replace('_', ' ').title()
            ax.plot(angles, values, 'o-', linewidth=2, label=engine_label, 
                   color=colors_radar[i % len(colors_radar)])
            ax.fill(angles, values, alpha=0.15, color=colors_radar[i % len(colors_radar)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('🎯 Biểu Đồ Radar - Top 5 Engines (Ground Truth)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)
        
        output_path2 = os.path.join(self.charts_dir, f"{output_prefix}_radar.png")
        plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Biểu đồ radar: {output_path2}")
        
        # === 3. Biểu đồ Heatmap - All metrics ===
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Chuẩn bị dữ liệu cho heatmap
        metrics_data = []
        for engine_key, engine_data in sorted_engines:
            metrics_data.append([
                engine_data.get('avg_f1_score', 0) * 100,
                engine_data.get('avg_precision', 0) * 100,
                engine_data.get('avg_recall', 0) * 100,
                engine_data.get('avg_char_accuracy', 0) * 100
            ])
        
        metrics_array = np.array(metrics_data)
        
        im = ax.imshow(metrics_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Thiết lập ticks
        ax.set_xticks(np.arange(len(categories)))
        ax.set_yticks(np.arange(len(engine_names)))
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_yticklabels(engine_names, fontsize=10)
        
        # Hiển thị giá trị
        for i in range(len(engine_names)):
            for j in range(len(categories)):
                text = ax.text(j, i, f'{metrics_array[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        ax.set_title('🔥 Heatmap - Tất Cả Metrics (Càng xanh càng tốt)', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        
        output_path3 = os.path.join(self.charts_dir, f"{output_prefix}_heatmap.png")
        plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Biểu đồ heatmap: {output_path3}")
        
        return {
            'comparison': output_path1,
            'radar': output_path2,
            'heatmap': output_path3
        }
    
    def create_detailed_engine_comparison(self, evaluation_json_path, output_prefix="engine_comparison"):
        """
        Tạo biểu đồ so sánh chi tiết từng engine với tất cả metrics
        
        Args:
            evaluation_json_path: Đường dẫn đến file evaluation_report_*.json
            output_prefix: Tiền tố cho tên file output
        
        Returns:
            Dict chứa đường dẫn các file chart
        """
        print(f"📊 Tạo biểu đồ so sánh chi tiết engines...")
        
        # Đọc dữ liệu evaluation
        try:
            if not os.path.isabs(evaluation_json_path):
                eval_path = os.path.join(self.results_dir, evaluation_json_path)
                if not os.path.exists(eval_path):
                    eval_path = evaluation_json_path
            else:
                eval_path = evaluation_json_path
            
            with open(eval_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ Đã đọc dữ liệu evaluation")
        except Exception as e:
            print(f"❌ Lỗi đọc file: {str(e)}")
            return None
        
        summary = data.get('results', {}).get('summary', {})
        if not summary:
            # Thử lấy summary ở root level
            summary = data.get('summary', {})
        
        if not summary:
            print("❌ Không có dữ liệu summary trong file evaluation")
            return None
        
        # Sắp xếp theo F1-score
        sorted_engines = sorted(summary.items(), key=lambda x: x[1].get('avg_f1_score', 0), reverse=True)
        
        engine_names = [e[0].replace('_', ' ').title() for e in sorted_engines]
        f1_scores = [e[1].get('avg_f1_score', 0) * 100 for e in sorted_engines]
        precisions = [e[1].get('avg_precision', 0) * 100 for e in sorted_engines]
        recalls = [e[1].get('avg_recall', 0) * 100 for e in sorted_engines]
        char_accs = [e[1].get('avg_char_accuracy', 0) * 100 for e in sorted_engines]
        times = [e[1].get('avg_processing_time', 0) for e in sorted_engines]
        
        # === 1. Biểu đồ cột chi tiết - 5 metrics trong 1 chart ===
        fig, ax = plt.subplots(figsize=(16, 10))
        
        x = np.arange(len(engine_names))
        width = 0.15  # Độ rộng mỗi cột
        
        # Vẽ 4 metrics chính (F1, Precision, Recall, CharAcc) - scale 0-100%
        bars1 = ax.bar(x - 2*width, f1_scores, width, label='F1-Score', color='#3498db', alpha=0.9)
        bars2 = ax.bar(x - width, precisions, width, label='Precision', color='#2ecc71', alpha=0.9)
        bars3 = ax.bar(x, recalls, width, label='Recall', color='#f39c12', alpha=0.9)
        bars4 = ax.bar(x + width, char_accs, width, label='Char Accuracy', color='#e74c3c', alpha=0.9)
        
        # Tạo secondary axis cho time (scale khác)
        ax2 = ax.twinx()
        bars5 = ax2.bar(x + 2*width, times, width, label='Time (s)', color='#9b59b6', alpha=0.9)
        
        # Thiết lập trục chính (Accuracy %)
        ax.set_xlabel('OCR Engines', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_title('📊 CHI TIẾT SO SÁNH CÁC METRICS GIỮA CÁC OCR ENGINES\n(Dựa trên Ground Truth)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(engine_names, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper left', fontsize=10)
        
        # Thiết lập trục phụ (Time)
        ax2.set_ylabel('Processing Time (seconds)', fontweight='bold', fontsize=12)
        ax2.set_ylim(0, max(times) * 1.2 if times else 10)
        ax2.legend(loc='upper right', fontsize=10)
        
        # Thêm giá trị trên mỗi cột
        def add_value_labels(bars, ax_obj, is_time=False):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    if is_time:
                        label = f'{height:.1f}s'
                    else:
                        label = f'{height:.1f}%'
                    ax_obj.text(bar.get_x() + bar.get_width()/2., height,
                              label, ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        add_value_labels(bars1, ax)
        add_value_labels(bars2, ax)
        add_value_labels(bars3, ax)
        add_value_labels(bars4, ax)
        add_value_labels(bars5, ax2, is_time=True)
        
        plt.tight_layout()
        
        output_path1 = os.path.join(self.charts_dir, f"{output_prefix}_detailed_bars.png")
        plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Biểu đồ cột chi tiết: {output_path1}")
        
        # === 2. Biểu đồ so sánh từng cặp metrics ===
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 SO SÁNH CHI TIẾT TỪNG METRICS', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(engine_names)))
        
        # Chart 1: F1-Score
        ax1.barh(engine_names, f1_scores, color=colors, alpha=0.8)
        ax1.set_xlabel('F1-Score (%)', fontweight='bold')
        ax1.set_title('🎯 F1-Score (Harmonic Mean of Precision & Recall)', fontweight='bold')
        ax1.set_xlim(0, 100)
        ax1.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(f1_scores):
            if v > 0:
                ax1.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
        
        # Chart 2: Precision & Recall
        x_pos = np.arange(len(engine_names))
        width = 0.35
        ax2.barh(x_pos - width/2, precisions, width, label='Precision', color='#2ecc71', alpha=0.8)
        ax2.barh(x_pos + width/2, recalls, width, label='Recall', color='#f39c12', alpha=0.8)
        ax2.set_yticks(x_pos)
        ax2.set_yticklabels(engine_names)
        ax2.set_xlabel('Percentage (%)', fontweight='bold')
        ax2.set_title('⚖️ Precision vs Recall', fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Chart 3: Character Accuracy
        ax3.barh(engine_names, char_accs, color=colors, alpha=0.8)
        ax3.set_xlabel('Character Accuracy (%)', fontweight='bold')
        ax3.set_title('📝 Character-Level Accuracy (Levenshtein Distance)', fontweight='bold')
        ax3.set_xlim(0, 100)
        ax3.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(char_accs):
            if v > 0:
                ax3.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
        
        # Chart 4: Processing Time
        ax4.barh(engine_names, times, color=colors, alpha=0.8)
        ax4.set_xlabel('Time (seconds)', fontweight='bold')
        ax4.set_title('⏱️ Average Processing Time', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(times):
            if v > 0:
                ax4.text(v + 0.5, i, f'{v:.2f}s', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        output_path2 = os.path.join(self.charts_dir, f"{output_prefix}_metrics_grid.png")
        plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Biểu đồ lưới metrics: {output_path2}")
        
        # === 3. Scatter Plot: Speed vs Accuracy ===
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(times, f1_scores, s=500, c=range(len(engine_names)), 
                           cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
        
        # Thêm tên engine
        for i, name in enumerate(engine_names):
            ax.annotate(name, (times[i], f1_scores[i]), 
                       fontsize=10, fontweight='bold',
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('⏱️ Processing Time (seconds)', fontweight='bold', fontsize=12)
        ax.set_ylabel('🎯 F1-Score (%)', fontweight='bold', fontsize=12)
        ax.set_title('⚡ TRADE-OFF: SPEED vs ACCURACY\n(Góc trên bên trái = Tốt nhất)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Thêm đường phân vùng
        median_time = np.median(times)
        median_f1 = np.median(f1_scores)
        ax.axvline(median_time, color='red', linestyle='--', alpha=0.5, label='Median Time')
        ax.axhline(median_f1, color='blue', linestyle='--', alpha=0.5, label='Median F1')
        ax.legend()
        
        # Thêm chú thích vùng
        ax.text(0.02, 0.98, '🏆 BEST ZONE\n(Fast & Accurate)', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        
        output_path3 = os.path.join(self.charts_dir, f"{output_prefix}_speed_vs_accuracy.png")
        plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Biểu đồ Speed vs Accuracy: {output_path3}")
        
        # === 4. Comprehensive Table Chart ===
        fig, ax = plt.subplots(figsize=(14, len(engine_names) * 0.8 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Tạo bảng dữ liệu
        table_data = []
        headers = ['Rank', 'Engine', 'F1-Score', 'Precision', 'Recall', 'Char Acc', 'Time (s)']
        
        for i, (engine_key, engine_data) in enumerate(sorted_engines, 1):
            emoji = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f'#{i}'
            row = [
                emoji,
                engine_key.replace('_', ' ').title(),
                f"{engine_data.get('avg_f1_score', 0) * 100:.2f}%",
                f"{engine_data.get('avg_precision', 0) * 100:.2f}%",
                f"{engine_data.get('avg_recall', 0) * 100:.2f}%",
                f"{engine_data.get('avg_char_accuracy', 0) * 100:.2f}%",
                f"{engine_data.get('avg_processing_time', 0):.2f}s"
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.08, 0.25, 0.12, 0.12, 0.12, 0.12, 0.12])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Tô màu header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Tô màu rows xen kẽ
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                
                # Highlight top 3
                if i <= 3:
                    table[(i, j)].set_facecolor('#ffffcc')
        
        plt.title('📋 BẢNG SO SÁNH ĐẦY ĐỦ CÁC OCR ENGINES\n(Sắp xếp theo F1-Score)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        output_path4 = os.path.join(self.charts_dir, f"{output_prefix}_table.png")
        plt.savefig(output_path4, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Biểu đồ bảng: {output_path4}")
        
        return {
            'detailed_bars': output_path1,
            'metrics_grid': output_path2,
            'speed_vs_accuracy': output_path3,
            'table': output_path4
        }
    
    def create_all_charts_from_json(self, json_file_path, output_prefix="ocr_analysis"):
        """Tạo tất cả biểu đồ từ file JSON"""
        print(f"🎨 Tạo tất cả biểu đồ từ {json_file_path}...")
        
        results = {
            'grouped_bar': None,
            'radar': None,
            'bubble': None
        }
        
        try:
            # Tạo biểu đồ cột nhóm
            results['grouped_bar'] = self.create_grouped_bar_chart_from_json(
                json_file_path, f"{output_prefix}_grouped_bar"
            )
            
            # Tạo biểu đồ radar
            results['radar'] = self.create_radar_chart_from_json(
                json_file_path, f"{output_prefix}_radar"
            )
            
            # Tạo biểu đồ bong bóng
            results['bubble'] = self.create_bubble_chart_from_json(
                json_file_path, f"{output_prefix}_bubble"
            )
            
            print(f"✅ Tất cả biểu đồ đã được tạo với prefix: {output_prefix}")
            return results
            
        except Exception as e:
            print(f"❌ Lỗi tạo biểu đồ: {str(e)}")
            import traceback
            traceback.print_exc()
            return results

def main():
    """Test function"""
    tool = JSONOCRVisualizationTool()
    
    # Tìm file JSON comparison report mới nhất trong thư mục Json
    json_files = [f for f in os.listdir(tool.json_dir) if f.startswith('comparison_report_') and f.endswith('.json')]
    if not json_files:
        print(f"❌ Không tìm thấy file comparison report JSON trong {tool.json_dir}")
        return
    
    # Sử dụng file mới nhất
    latest_file = max([os.path.join(tool.json_dir, f) for f in json_files], key=os.path.getctime)
    latest_filename = os.path.basename(latest_file)
    print(f"📁 Sử dụng file: {latest_filename}")
    
    # Tạo tất cả biểu đồ
    results = tool.create_all_charts_from_json(latest_filename, "json_analysis")
    
    print("\n🎯 KẾT QUẢ:")
    for chart_type, file_path in results.items():
        if file_path:
            print(f"✅ {chart_type}: {os.path.basename(file_path)}")
        else:
            print(f"❌ {chart_type}: Không tạo được")

if __name__ == "__main__":
    main()