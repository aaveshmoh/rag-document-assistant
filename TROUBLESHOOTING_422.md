# 422 Unprocessable Content Error - Troubleshooting Guide

## What Does 422 Mean?

A 422 (Unprocessable Entity) error means FastAPI couldn't validate your request body against the expected `QARequest` model. The request was syntactically valid JSON, but didn't match the required structure.

## Common Causes & Solutions

### 1. ❌ Missing `question` Field

**Problem:**

```json
{}
```

**Solution:**

```json
{
  "question": "What are the minimum loan requirements?"
}
```

---

### 2. ❌ Wrong Field Name

**Problem:**

```json
{
  "text": "What are the minimum loan requirements?"
}
```

**Solution:**

```json
{
  "question": "What are the minimum loan requirements?"
}
```

---

### 3. ❌ Wrong Data Type

**Problem:**

```json
{
  "question": 123
}
```

**Solution:**

```json
{
  "question": "123"
}
```

---

### 4. ❌ Missing Content-Type Header

**Problem:**

```bash
curl -X POST http://127.0.0.1:8000/qa \
  -d '{"question": "What are rates?"}'
```

**Solution:**

```bash
curl -X POST http://127.0.0.1:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What are rates?"}'
```

---

### 5. ❌ Extra Unknown Fields

**Problem:**

```json
{
  "question": "What are rates?",
  "extra_field": "This will cause an error"
}
```

**Solution:**

```json
{
  "question": "What are rates?"
}
```

---

## Correct Request Examples

### Using cURL

```bash
# Basic request
curl -X POST http://127.0.0.1:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the minimum loan requirements?"}'
```

### Using Python

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/qa",
    json={"question": "What are the minimum loan requirements?"},
    headers={"Content-Type": "application/json"}
)

print(response.json())
```

### Using JavaScript/Fetch API

```javascript
fetch("http://127.0.0.1:8000/qa", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    question: "What are the minimum loan requirements?",
  }),
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

### Using Postman

1. Set method to **POST**
2. Enter URL: `http://127.0.0.1:8000/qa`
3. Go to **Headers** tab
4. Add header: `Content-Type: application/json`
5. Go to **Body** tab
6. Select **raw** and **JSON**
7. Enter:

```json
{
  "question": "What are the minimum loan requirements?"
}
```

8. Click **Send**

---

## API Request/Response Contract

### Request Format

```
POST /qa HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "question": "string (required)"
}
```

### Success Response (200)

```json
{
  "question": "What are the minimum loan requirements?",
  "answer": "Based on the banking policy...",
  "sources": ["Commercial Loan Policy 2024"]
}
```

### Error Response (422)

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "question"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

---

## Debugging Steps

### Step 1: Check API is Running

```bash
curl http://127.0.0.1:8000/
```

Should return:

```json
{
  "status": "ok",
  "message": "Banking RAG QA API is running.",
  "ollama": "connected or disconnected"
}
```

### Step 2: Validate JSON Syntax

Use an online JSON validator: https://jsonlint.com/

### Step 3: Use Test Script

```bash
cd c:\codebase\ai-learning\RAG-Design-QA
python test_api.py
```

### Step 4: Check Request Headers

```bash
curl -v -X POST http://127.0.0.1:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
```

The `-v` flag shows all headers and request details.

---

## FastAPI Interactive Documentation

Once the API is running, visit:

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

These provide interactive API documentation where you can:

- Test endpoints directly
- See request/response schemas
- View validation errors in real-time

---

## Common Environment Issues

### Issue: "Python was not found"

**Solution**: Use Python explicitly from your virtual environment or add Python to PATH

```bash
# Using Python directly
C:\path\to\python.exe -m uvicorn api:app --reload

# Or using venv
.\venv\Scripts\python -m uvicorn api:app --reload
```

### Issue: "Module not found: api"

**Solution**: Make sure you're in the correct directory

```bash
cd c:\codebase\ai-learning\RAG-Design-QA
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

### Issue: Port already in use (Address already in use)

**Solution**: Use a different port or kill the existing process

```bash
# Try a different port
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8001

# Or find and kill the existing process
# On Windows (PowerShell):
Get-Process -Name python | Stop-Process -Force
```

---

## Quick Checklist

Before submitting a request, verify:

- ✅ API is running (`curl http://127.0.0.1:8000/`)
- ✅ Request body has valid JSON
- ✅ `question` field is present and is a string
- ✅ No extra unknown fields in request
- ✅ `Content-Type: application/json` header is set
- ✅ Question text is not empty
- ✅ No trailing/leading special characters in JSON

---

## Support

If you're still getting 422 errors:

1. Run the test script: `python test_api.py`
2. Check FastAPI docs at: `http://127.0.0.1:8000/docs`
3. Review server logs for detailed error messages
4. Verify all fields in the JSON payload match the schema exactly
