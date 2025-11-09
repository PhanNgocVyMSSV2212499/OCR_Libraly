"""
GOCR Module for Vietnamese Text Recognition
Smart wrapper with Docker fallback - Fixed Version
"""

import cv2
import numpy as np
import time
import subprocess
import os
import tempfile
import shutil
import re

class GOCRProcessor:
    def __init__(self):
        """
        Khởi tạo GOCR processor với Docker fallback
        """
        print("Đang khởi tạo GOCR...")
        
        # Kiểm tra Docker GOCR trước
        self.use_docker = self._check_docker_gocr()
        self.gocr_path = None
        
        if self.use_docker:
            print("✓ Tìm thấy GOCR Docker container")
            print("✓ GOCR (Docker) đã được khởi tạo")
        else:
            # Fallback tìm GOCR native
            self.gocr_path = self._find_gocr_executable()
            if self.gocr_path:
                print(f"✓ Tìm thấy GOCR native tại: {self.gocr_path}")
                print("✓ GOCR (Native) đã được khởi tạo")
            else:
                print("⚠️ GOCR không được tìm thấy, sử dụng fallback mode")
    
    def _check_docker_gocr(self):
        """
        Kiểm tra GOCR Docker container có sẵn không
        """
        try:
            # Thử các tên image có thể có
            image_names = [
                "gocr-test",
                "ocr_library-gocr-test", 
                "ocr_library_gocr-test",
                "ocr-library-gocr-test"
            ]
            
            for image_name in image_names:
                try:
                    result = subprocess.run([
                        "docker", "run", "--rm", 
                        image_name, 
                        "gocr", "--help"
                    ], capture_output=True, text=True, timeout=15)
                    
                    if result.returncode == 0:
                        self.docker_image = image_name
                        print(f"✓ Tìm thấy GOCR Docker image: {image_name}")
                        return True
                except:
                    continue
                    
            return False
        except Exception as e:
            print(f"⚠️ Không thể kiểm tra Docker: {str(e)}")
            return False
    
    def _find_gocr_executable(self):
        """Tìm GOCR executable"""
        # Các đường dẫn có thể có GOCR
        possible_paths = [
            'gocr',  # In PATH
            '/usr/bin/gocr',  # Linux standard
            '/usr/local/bin/gocr',  # Linux local
            'C:\\gocr\\gocr.exe',  # Windows manual install
            'C:\\msys64\\usr\\bin\\gocr.exe',  # MSYS2
            'C:\\Program Files\\GOCR\\gocr.exe',  # Windows standard
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 or 'gocr' in result.stdout.lower():
                    return path
            except:
                continue
        
        return None
    
    def detect_text(self, image_path):
        """
        Nhận dạng văn bản trong ảnh - Interface thống nhất với các engine khác
        
        Args:
            image_path: Đường dẫn đến ảnh
            
        Returns:
            Dictionary với format chuẩn
        """
        try:
            # Sử dụng method process_image hiện có
            result = self.process_image(image_path)
            
            # Chuyển đổi format để thống nhất với các engine khác
            if result.get('success', False):
                return {
                    'status': 'success',
                    'texts': [],  # GOCR không trả về texts array chi tiết
                    'full_text': result.get('text', ''),
                    'engine': 'GOCR',
                    'total_detections': result.get('word_count', 0),
                    'processing_time': result.get('processing_time', 0)
                }
            else:
                return {
                    'status': 'error',
                    'message': result.get('error', 'Unknown error'),
                    'texts': [],
                    'engine': 'GOCR'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'texts': [],
                'engine': 'GOCR'
            }

    def process_image(self, image_path):
        """
        Xử lý ảnh bằng GOCR (Docker hoặc Native)
        
        Args:
            image_path (str): Đường dẫn đến ảnh
            
        Returns:
            dict: Kết quả OCR
        """
        start_time = time.time()
        
        try:
            if self.use_docker:
                return self._process_with_docker(image_path, start_time)
            elif self.gocr_path:
                return self._process_with_native(image_path, start_time)
            else:
                return {
                    'success': False,
                    'error': 'GOCR không khả dụng. Cần cài đặt GOCR hoặc Docker.',
                    'text': '',
                    'confidence': 0,
                    'word_count': 0,
                    'processing_time': time.time() - start_time,
                    'engine': 'GOCR (Not Available)'
                }
        except Exception as e:
            return {
                'success': False,
                'error': f'GOCR error: {str(e)}',
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'processing_time': time.time() - start_time,
                'engine': 'GOCR (Error)'
            }
    
    def _process_with_docker(self, image_path, start_time):
        """
        Xử lý ảnh bằng GOCR Docker - Cải thiện với multiple preprocessing
        """
        temp_dir = None
        try:
            # Tạo thư mục tạm cho Docker
            temp_dir = tempfile.mkdtemp()
            print(f"🗂️ Created temp directory: {temp_dir}")
            
            # Copy ảnh gốc
            temp_image_name = "gocr_input.jpg"
            temp_image_path = os.path.join(temp_dir, temp_image_name)
            shutil.copy2(image_path, temp_image_path)
            print(f"📋 Copied image to: {temp_image_path}")
            
            # Thử multiple preprocessing techniques
            best_result = None
            best_confidence = 0
            
            preprocessing_methods = [
                'standard',
                'high_contrast', 
                'denoised',
                'enhanced'
            ]
            
            for method in preprocessing_methods:
                try:
                    processed_path = self._preprocess_for_docker(temp_image_path, method)
                    
                    # Verify file exists before Docker call
                    if not os.path.exists(processed_path):
                        print(f"❌ Processed file not found: {processed_path}")
                        continue
                        
                    print(f"✅ Processed file exists: {processed_path} ({os.path.getsize(processed_path)} bytes)")
                    
                    # Chạy GOCR trong Docker với tham số tối ưu
                    docker_input_path = f"/tmp/gocr/{os.path.basename(processed_path)}"
                    
                    cmd = [
                        "docker", "run", "--rm",
                        "-v", f"{temp_dir}:/tmp/gocr",
                        self.docker_image,
                        "gocr", 
                        "-i", docker_input_path,
                        "-f", "ASCII"
                    ]
                    
                    print(f"🔧 Running GOCR with method: {method}")
                    print(f"📂 Docker command: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    print(f"🔍 GOCR return code: {result.returncode}")
                    if result.stdout:
                        print(f"📝 GOCR stdout: {result.stdout[:100]}...")
                    if result.stderr:
                        print(f"⚠️ GOCR stderr: {result.stderr[:100]}...")
                    
                    if result.returncode == 0 and result.stdout.strip():
                        text = result.stdout.strip()
                        
                        if text:  # Chỉ xử lý nếu có text
                            # Enhanced text cleaning
                            cleaned_text = self._enhanced_clean_gocr_output(text)
                            
                            # Improved confidence calculation
                            confidence = self._calculate_smart_confidence(cleaned_text)
                            
                            # Chọn kết quả tốt nhất
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_result = {
                                    'text': cleaned_text,
                                    'confidence': confidence,
                                    'method': method
                                }
                    
                except Exception as e:
                    print(f"Lỗi preprocessing method {method}: {str(e)}")
                    continue
            
            # Dọn dẹp
            shutil.rmtree(temp_dir)
            
            processing_time = time.time() - start_time
            
            if best_result:
                word_count = len(best_result['text'].split()) if best_result['text'] else 0
                
                return {
                    'success': True,
                    'text': best_result['text'] if best_result['text'] else "No text detected",
                    'confidence': best_result['confidence'],
                    'word_count': word_count,
                    'processing_time': processing_time,
                    'engine': f"GOCR (Docker-{best_result['method']})",
                    'docker_image': self.docker_image
                }
            else:
                return {
                    'success': False,
                    'error': 'Không thể nhận dạng text với tất cả phương pháp preprocessing',
                    'text': '',
                    'confidence': 0,
                    'word_count': 0,
                    'processing_time': processing_time,
                    'engine': 'GOCR (Docker No Text)'
                }
            
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
                print(f"🗑️ Cleaned up temp directory: {temp_dir}")
            except:
                pass
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Docker GOCR timeout (>60s)',
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'processing_time': time.time() - start_time,
                'engine': 'GOCR (Docker Timeout)'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Docker GOCR exception: {str(e)}',
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'processing_time': time.time() - start_time,
                'engine': 'GOCR (Docker Exception)'
            }
    
    def _process_with_native(self, image_path, start_time):
        """
        Xử lý ảnh bằng GOCR native
        """
        try:
            # Tiền xử lý ảnh
            processed_image_path = self._preprocess_for_gocr(image_path)
            if not processed_image_path:
                raise Exception("Không thể tiền xử lý ảnh")
            
            # Chạy GOCR với tham số tối ưu
            cmd = [
                self.gocr_path,
                "-i", processed_image_path,
                "-f", "ASCII",
                "-l", "2",  # Layout analysis level 2
                "-a", "95", # Accuracy 95%
                "-m", "256" # Recognition mode 256
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Dọn dẹp file tạm
            try:
                os.unlink(processed_image_path)
            except:
                pass
            
            processing_time = time.time() - start_time
            
            if result.returncode == 0:
                text = result.stdout.strip()
                cleaned_text = self._enhanced_clean_gocr_output(text)
                confidence = self._calculate_smart_confidence(cleaned_text)
                word_count = len(cleaned_text.split()) if cleaned_text else 0
                
                return {
                    'success': True,
                    'text': cleaned_text if cleaned_text else "No text detected",
                    'confidence': confidence,
                    'word_count': word_count,
                    'processing_time': processing_time,
                    'engine': 'GOCR (Native)'
                }
            else:
                error_msg = result.stderr.strip() if result.stderr else "GOCR processing failed"
                return {
                    'success': False,
                    'error': f'GOCR error: {error_msg}',
                    'text': '',
                    'confidence': 0,
                    'word_count': 0,
                    'processing_time': processing_time,
                    'engine': 'GOCR (Native Error)'
                }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'GOCR timeout (>30s)',
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'processing_time': time.time() - start_time,
                'engine': 'GOCR (Native Timeout)'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'GOCR exception: {str(e)}',
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'processing_time': time.time() - start_time,
                'engine': 'GOCR (Native Exception)'
            }
    
    def _preprocess_for_docker(self, image_path, method='standard'):
        """
        Tiền xử lý ảnh cho Docker GOCR với nhiều phương pháp
        """
        try:
            # Đọc ảnh
            image = cv2.imread(image_path)
            if image is None:
                return image_path
            
            # Chuyển sang grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if method == 'standard':
                # Standard preprocessing
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            elif method == 'high_contrast':
                # High contrast
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                # Sharpen
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                gray = cv2.filter2D(gray, -1, kernel)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            elif method == 'denoised':
                # Denoise first
                gray = cv2.fastNlMeansDenoising(gray)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            elif method == 'enhanced':
                # Enhanced preprocessing
                gray = cv2.fastNlMeansDenoising(gray)
                # Morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Lưu ảnh đã xử lý
            method_suffix = f"_{method}" if method != 'standard' else ""
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            processed_name = f"{base_name}{method_suffix}_processed.pbm"
            processed_path = os.path.join(os.path.dirname(image_path), processed_name)
            cv2.imwrite(processed_path, binary)
            
            print(f"🖼️ Preprocessed image saved: {processed_path}")
            return processed_path
            
        except Exception as e:
            print(f"Lỗi tiền xử lý {method}: {str(e)}")
            return image_path
    
    def _preprocess_for_gocr(self, image_path):
        """
        Tiền xử lý ảnh cho GOCR native
        """
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhanced preprocessing for native GOCR
            gray = cv2.fastNlMeansDenoising(gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Save as PBM
            temp_file = tempfile.NamedTemporaryFile(suffix='.pbm', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            cv2.imwrite(temp_path, binary)
            
            return temp_path
        except:
            return None
    
    def _enhanced_clean_gocr_output(self, text):
        """
        Enhanced cleaning cho GOCR output
        """
        if not text:
            return ""
        
        # 1. Replace common GOCR misrecognitions
        replacements = {
            # Common character misrecognitions
            'rn': 'm',
            'vv': 'w', 
            'ii': 'll',
            '1l': 'll',
            '0O': 'OO',
            '5S': 'SS',
            '6G': 'GG',
            'cl': 'd',
            'o0': 'oo',
            'nn': 'mm',
            # Vietnamese specific
            'â': 'ă',  # Sometimes mixed up
            'ơ': 'ư',  # Sometimes mixed up
        }
        
        cleaned_text = text
        for old, new in replacements.items():
            cleaned_text = cleaned_text.replace(old, new)
        
        # 2. Remove excessive spaces and weird characters
        cleaned_text = re.sub(r'[^\w\s\-.,!?:;()áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ]', ' ', cleaned_text, flags=re.IGNORECASE)
        
        # 3. Fix spacing
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        # 4. Fix common Vietnamese words
        vietnamese_fixes = {
            'va': 'và',
            'co': 'có',
            'thi': 'thì',
            'nhu': 'như',
            'cho': 'cho',
            'nha': 'nhà',
            'cua': 'của'
        }
        
        words = cleaned_text.split()
        for i, word in enumerate(words):
            if word.lower() in vietnamese_fixes:
                words[i] = vietnamese_fixes[word.lower()]
        
        return ' '.join(words)
    
    def _calculate_smart_confidence(self, text):
        """
        Tính confidence thông minh dựa trên chất lượng text thực tế
        """
        if not text:
            return 0.0
        
        score = 0.1  # Lower base score
        
        # 1. Vietnamese words recognition (tăng trọng số)
        vietnamese_words = [
            'và', 'của', 'cho', 'nhà', 'với', 'trong', 'một', 'có', 'người', 
            'được', 'từ', 'họ', 'năm', 'tại', 'về', 'đây', 'đó', 'sẽ', 'sau', 'nó',
            'là', 'không', 'này', 'các', 'theo', 'những', 'thì', 'giáo', 'dục',
            'nguyễn', 'thành', 'phạm', 'ngọc', 'lan', 'trần', 'lê', 'hoa', 'ngữ', 'văn'
        ]
        words = text.lower().split()
        if len(words) > 0:
            vietnamese_count = sum(1 for word in words if word in vietnamese_words)
            vietnamese_ratio = vietnamese_count / len(words)
            score += vietnamese_ratio * 0.4  # Tăng trọng số từ 0.2 → 0.4
        
        # 2. Readable words (từ có ít nhất 3 ký tự liên tiếp là chữ)
        readable_words = re.findall(r'[a-zA-ZáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ]{3,}', text)
        if len(words) > 0:
            readable_ratio = len(readable_words) / len(words)
            score += readable_ratio * 0.3  # Thưởng từ đọc được
        
        # 3. Phạt nặng ký tự lạ 
        noise_chars = len(re.findall(r'[_\?]|\(\?\)', text))
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0:
            noise_ratio = noise_chars / total_chars
            score -= noise_ratio * 0.5  # Phạt nặng hơn
        
        # 4. Phạt text có quá nhiều số và ký tự đơn lẻ
        isolated_chars = len(re.findall(r'\b[a-zA-Z0-9]\b', text))  # Ký tự đơn lẻ
        if len(words) > 0:
            isolated_ratio = isolated_chars / len(words)
            score -= isolated_ratio * 0.2
        
        # 5. Thưởng cấu trúc văn bản bình thường
        if re.search(r'[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]', text):
            score += 0.1  # Có chữ hoa
        
        # 6. Text length (giảm trọng số)
        if len(text) > 20:
            score += 0.1  # Chỉ thưởng nhẹ cho text dài
        
        # Normalize to 0-1
        return max(0.0, min(1.0, score))

# Test function
def test_gocr():
    """
    Test GOCR processor
    """
    print("🧪 TESTING IMPROVED GOCR PROCESSOR")
    print("=" * 40)
    
    processor = GOCRProcessor()
    
    # Test với ảnh
    import os
    test_image = None
    if os.path.exists("../Bia_sach"):
        for file in os.listdir("../Bia_sach"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_image = os.path.join("../Bia_sach", file)
                break
    
    if test_image:
        print(f"🖼️  Test với: {test_image}")
        result = processor.detect_text(test_image)
        
        if result['status'] == 'success':
            print(f"✅ Success: {result['full_text'][:100]}...")
            print(f"📊 Words: {result['total_detections']}")
        else:
            print(f"❌ Error: {result['message']}")
    else:
        print("❌ Không tìm thấy ảnh test")

if __name__ == "__main__":
    test_gocr()