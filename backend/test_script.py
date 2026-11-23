import requests

url = 'http://127.0.0.1:8000/api/v1/analyze/field'
file_path = 'test_images/healthy_corn.jpg'

try:
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
        print(response.status_code)
        print(response.json())
except Exception as e:
    print(f'Error: {e}')
