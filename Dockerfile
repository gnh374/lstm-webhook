# Gunakan base image dengan TensorFlow
FROM python:3.9-slim

# Set working directory di container
WORKDIR /app

# Salin file proyek ke dalam container
COPY . .

# Install dependency
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port (sesuaikan jika pakai Flask/FastAPI)
EXPOSE 5000

# Jalankan aplikasi (ganti dengan perintah yang sesuai)
CMD ["python", "app.py"]
