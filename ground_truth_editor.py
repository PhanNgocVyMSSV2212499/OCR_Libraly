"""
Ground Truth Editor
Tool để thêm/sửa ground truth cho các ảnh
"""

import json
import os

class GroundTruthEditor:
    def __init__(self, ground_truth_file="ground_truth.json"):
        # Nếu ground_truth_file không phải là absolute path, tìm từ thư mục gốc project
        if not os.path.isabs(ground_truth_file):
            # Tìm thư mục gốc project
            base_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(base_dir, ground_truth_file)
            
            if os.path.exists(full_path):
                ground_truth_file = full_path
        
        self.ground_truth_file = ground_truth_file
        self.data = self.load_data()
    
    def load_data(self):
        """Load ground truth data"""
        if os.path.exists(self.ground_truth_file):
            with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {"images": []}
    
    def save_data(self):
        """Save ground truth data"""
        with open(self.ground_truth_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu vào {self.ground_truth_file}")
    
    def add_image(self, filename):
        """Thêm ground truth cho ảnh mới"""
        # Kiểm tra xem ảnh đã tồn tại chưa
        for img in self.data['images']:
            if img['filename'] == filename:
                print(f"⚠️ Ảnh {filename} đã tồn tại. Sử dụng edit_image() để sửa.")
                return
        
        print(f"\n{'='*60}")
        print(f"📝 THÊM GROUND TRUTH CHO: {filename}")
        print(f"{'='*60}")
        print("📄 Nhập TOÀN BỘ TEXT trên bìa sách:")
        print("(Gõ chính xác tất cả text bạn nhìn thấy)")
        print("(Nhập xong gõ Enter 2 lần)")
        print("="*60)
        
        lines = []
        empty_count = 0
        while empty_count < 2:
            line = input()
            if not line:
                empty_count += 1
            else:
                empty_count = 0
                lines.append(line)
        
        all_text = ' '.join(lines).strip()
        
        if not all_text:
            print("❌ Không có text nào được nhập!")
            return
        
        entry = {
            "filename": filename,
            "ground_truth": all_text
        }
        
        self.data['images'].append(entry)
        self.save_data()
        
        print("\n✅ Đã thêm ground truth cho", filename)
    
    def edit_image(self, filename):
        """Chỉnh sửa ground truth cho ảnh đã tồn tại"""
        entry = None
        for img in self.data['images']:
            if img['filename'] == filename:
                entry = img
                break
        
        if not entry:
            print(f"❌ Không tìm thấy {filename}")
            return
        
        print(f"\n{'='*60}")
        print(f"✏️  CHỈNH SỬA GROUND TRUTH: {filename}")
        print(f"{'='*60}")
        print("Text hiện tại:")
        print(entry['ground_truth'])
        print(f"\n{'='*60}")
        print("Nhập lại TOÀN BỘ TEXT (Enter 2 lần để giữ nguyên):")
        
        lines = []
        empty_count = 0
        while empty_count < 2:
            line = input()
            if not line:
                empty_count += 1
            else:
                empty_count = 0
                lines.append(line)
        
        if lines:
            entry['ground_truth'] = ' '.join(lines).strip()
            self.save_data()
            print("\n✅ Đã cập nhật ground truth cho", filename)
        else:
            print("\n⚠️ Giữ nguyên ground truth")
    
    def list_images(self):
        """Liệt kê tất cả ảnh có ground truth"""
        print(f"\n{'='*60}")
        print(f"📋 DANH SÁCH GROUND TRUTH ({len(self.data['images'])} ảnh)")
        print(f"{'='*60}")
        
        for i, img in enumerate(self.data['images'], 1):
            text_preview = img['ground_truth'][:50] + '...' if len(img['ground_truth']) > 50 else img['ground_truth']
            print(f"{i}. {img['filename']}")
            print(f"   Text: {text_preview}")
    
    def remove_image(self, filename):
        """Xóa ground truth của ảnh"""
        self.data['images'] = [img for img in self.data['images'] if img['filename'] != filename]
        self.save_data()
        print(f"✅ Đã xóa ground truth của {filename}")


def main():
    editor = GroundTruthEditor("ground_truth.json")
    
    while True:
        print("\n" + "="*60)
        print("🔧 GROUND TRUTH EDITOR")
        print("="*60)
        print("1. Thêm ground truth cho ảnh mới")
        print("2. Chỉnh sửa ground truth")
        print("3. Xem danh sách")
        print("4. Xóa ground truth")
        print("0. Thoát")
        print("="*60)
        
        choice = input("Chọn (0-4): ").strip()
        
        if choice == "1":
            filename = input("Tên file ảnh: ").strip()
            if filename:
                editor.add_image(filename)
        
        elif choice == "2":
            editor.list_images()
            filename = input("\nTên file cần sửa: ").strip()
            if filename:
                editor.edit_image(filename)
        
        elif choice == "3":
            editor.list_images()
        
        elif choice == "4":
            editor.list_images()
            filename = input("\nTên file cần xóa: ").strip()
            if filename:
                confirm = input(f"Xác nhận xóa {filename}? (y/n): ").strip().lower()
                if confirm == 'y':
                    editor.remove_image(filename)
        
        elif choice == "0":
            print("👋 Tạm biệt!")
            break


if __name__ == "__main__":
    main()
