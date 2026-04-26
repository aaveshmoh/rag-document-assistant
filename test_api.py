#!/usr/bin/env python3
"""
Test script for the RAG QA API
Tests the /qa endpoint with proper request format
"""

import requests
import json

def test_qa_endpoint():
    """Test the /qa POST endpoint with proper JSON request"""
    
    api_url = "http://127.0.0.1:8000/qa"
    
    # Correct request format - ask about the PDF content (CV/Resume)
    request_data = {
        "question": "What are Mohammad Aavesh's technical skills and experience?"
    }
    
    print("=" * 60)
    print("Testing RAG QA API")
    print("=" * 60)
    print(f"\nURL: {api_url}")
    print(f"Request Body: {json.dumps(request_data, indent=2)}")
    print(f"Content-Type: application/json")
    
    try:
        response = requests.post(
            api_url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"\nResponse Body:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print("\n✅ SUCCESS - API is working correctly!")
        elif response.status_code == 422:
            print("\n❌ ERROR 422 - Validation Error")
            print("Possible causes:")
            print("  1. Missing 'question' field in request body")
            print("  2. Wrong Content-Type header")
            print("  3. Malformed JSON")
            print("  4. Extra unknown fields in request")
        else:
            print(f"\n⚠️  Unexpected status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("Make sure the API is running at http://127.0.0.1:8000")
        print("\nTo start the API, run:")
        print("  python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def test_health_endpoint():
    """Test the health check endpoint"""
    
    api_url = "http://127.0.0.1:8000/"
    
    print("\n" + "=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)
    print(f"URL: {api_url}")
    
    try:
        response = requests.get(api_url, timeout=10)
        print(f"Response Status: {response.status_code}")
        print(f"Response Body:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print("\n✅ Health check passed!")
        else:
            print(f"\n⚠️  Unexpected status: {response.status_code}")
            
    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ Connection Error: API is not running")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    print("\nTesting API endpoints...\n")
    test_health_endpoint()
    print("\n")
    test_qa_endpoint()
    print("\n" + "=" * 60)
