# 📌 استفاده از نسخه slim برای کاهش حجم
FROM python:3.10-slim

# 🧱 نصب وابستگی‌های سیستمی لازم برای OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 🗂 مسیر کاری داخل کانتینر
WORKDIR /app

# 📝 کپی کردن فایل‌های پروژه
COPY . /app

# 🔧 نصب وابستگی‌های پایتون
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 🌐 باز کردن پورت مورد استفاده
EXPOSE 8000

# 🚀 اجرای اپ با uvicorn (در محیط واقعی بهتر با gunicorn اجرا بشه)
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port", "8000"]
