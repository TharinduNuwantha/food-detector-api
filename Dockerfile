FROM python:3.10-slim

# System dependencies ස්ථාපනය කිරීම
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user කෙනෙකු සෑදීම (UID 10014)
RUN useradd -m -u 10014 appuser

WORKDIR /app

# Dependencies ස්ථාපනය කිරීම
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project files copy කර අදාළ userට ownership ලබා දීම
COPY --chown=appuser:appuser . .

# Non-root user බවට මාරු වීම
USER 10014

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]