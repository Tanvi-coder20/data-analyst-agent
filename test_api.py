import requests
import os

url = "http://127.0.0.1:8000/api/"
base = r"C:\Users\TANVI SUNDARKAR\PycharmProjects\data-analyst-agent"

files = [
    ("files", ("questions.txt", open(os.path.join(base, "sample_questions.txt"), "rb"))),
    ("files", ("data.csv", open(os.path.join(base, "sample_questions.txt"), "rb")))
]

resp = requests.post(url, files=files)
print(resp.json())
