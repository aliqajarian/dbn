import requests
import json

def test_api():
    # Test data
    test_data = {
        "text": "This is a sample review text",
        "timestamp": "2024-01-01T00:00:00",
        "rating": 4.5
    }
    
    # Send request to API
    response = requests.post(
        "http://localhost:8000/predict",
        json=test_data
    )
    
    # Print response
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    test_api() 