# Hướng dẫn cài đặt và chạy dự án OCR_Library

Dự án này sử dụng Python và thư viện OCR để nhận dạng ký tự quang học (Optical Character Recognition).

## 🧩 1. Chuẩn bị môi trường

Trước khi bắt đầu, hãy chắc chắn rằng bạn đã cài:

- **Python 3.8+**
- **pip** (trình quản lý gói Python)

## ⚙️ 2. Cài đặt môi trường ảo

Mở terminal hoặc CMD tại thư mục chứa dự án, sau đó chạy:

```bash
cd OCR_Libraly
python -m venv venv
```

Kích hoạt môi trường ảo:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

## 📦 3. Cài đặt thư viện cần thiết

Sau khi môi trường ảo đã được kích hoạt, chạy lệnh:

```bash
pip install -r requirements.txt
```

Lệnh này sẽ tự động cài đặt tất cả các gói Python cần thiết được liệt kê trong file `requirements.txt`.

## 🚀 4. Chạy chương trình demo

Chạy chương trình nhận dạng ký tự mẫu:

```bash
python Demo/simple_ocr.py
```

Nếu mọi thứ được cài đặt đúng, chương trình sẽ thực thi và hiển thị kết quả OCR demo.

## 🧰 5. Ghi chú

- Nếu gặp lỗi liên quan đến **EasyOCR** hoặc **torch**, hãy đảm bảo rằng phiên bản Python và GPU driver tương thích.
- Bạn có thể chỉnh sửa đường dẫn hoặc hình ảnh trong `Demo/simple_ocr.py` để thử nhận dạng với ảnh riêng.

---

📅 _Cập nhật: Tháng 11, 2025_
