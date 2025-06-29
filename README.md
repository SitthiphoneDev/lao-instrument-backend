# How to run 
# Method 1: Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Method 2: Using python
python main.py

# Method 3: For development with auto-reload
uvicorn main:app --reload

