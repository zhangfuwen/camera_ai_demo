import subprocess
import sys
import os
import threading
import time
import webbrowser

def start_server():
    """Start the food detection server"""
    try:
        from food_detection_server import app
        print("Starting food detection server on http://localhost:5000")
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except ImportError as e:
        print(f"Error importing server: {e}")
        print("Make sure to install dependencies with: pip install -r requirements.txt")
        sys.exit(1)

def open_browser():
    """Open the browser after a delay to allow server to start"""
    time.sleep(3)  # Wait for server to start
    webbrowser.open('http://localhost:5001')

if __name__ == "__main__":
    print("Starting Camera AI Demo with Food Detection...")
    print("Installing dependencies...")
    
    # Install dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Open browser
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    print("Server started! Opening browser...")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
