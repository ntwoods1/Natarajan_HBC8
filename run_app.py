
import subprocess
import sys
import os
import webbrowser
import time
from threading import Timer

def open_browser():
    """Open browser after a short delay"""
    webbrowser.open('http://localhost:8501')

def main():
    """Launch the Streamlit app"""
    print("Starting Proteomics Data Analysis Platform...")
    print("The app will open in your default browser shortly.")
    print("Press Ctrl+C to stop the application.")
    
    # Open browser after 3 seconds
    Timer(3.0, open_browser).start()
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'main.py',
            '--server.port=8501',
            '--server.address=localhost',
            '--browser.gatherUsageStats=false'
        ])
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error starting app: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
