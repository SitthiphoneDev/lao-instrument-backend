 uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

#  Ngrok 
# choco install ngrok -y

 ngrok http --url=star-neat-stallion.ngrok-free.app 8000