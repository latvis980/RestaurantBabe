# ───── Base image ────────────────────────────────────────────
# FIXED: Updated to match Playwright 1.55.0
FROM mcr.microsoft.com/playwright/python:v1.55.0-noble

# ───── App setup ─────────────────────────────────────────────
WORKDIR /app

# 1. install Python deps (copy reqs first to leverage Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Create necessary directories
RUN mkdir -p data debug_logs

# 3. copy the rest of your code
COPY . .

# 4. default entry point — adjust to whatever starts your bot
CMD ["python", "-m", "main"]