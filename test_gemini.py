#!/usr/bin/env python3
"""
Test script for Gemini integration
"""
import os
from google import genai

def test_gemini_client():
    """Test Gemini client initialization and basic functionality"""
    try:
        # Initialize client
        api_key = os.getenv('GEMINI_API_KEY', 'sk-EplRjGkWQ9CwXK5w10936448E56a46BcB487EeE809C6Bd40')
        client = genai.Client(
            api_key=api_key,
            http_options={"base_url": "https://api.apiyi.com"}
        )
        print("✅ Gemini client initialized successfully")
        
        # Test basic content generation
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=["请用一句话介绍你自己。"]
        )
        
        print(f"✅ Gemini response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Gemini integration...")
    success = test_gemini_client()
    if success:
        print("✅ Gemini integration test passed")
    else:
        print("❌ Gemini integration test failed")
