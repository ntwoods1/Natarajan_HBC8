import sys
import os
import platform

def main():
    """Main launcher function"""
    print("Starting Proteomics Data Analysis Platform...")

    # Get the directory where the launcher is located
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        app_dir = os.path.dirname(sys.executable)
        # For PyInstaller, use _MEIPASS which points to the temporary extraction directory
        if hasattr(sys, '_MEIPASS'):
            main_py_path = os.path.join(sys._MEIPASS, 'main.py')
        else:
            # Fallback: look in the _internal directory where PyInstaller extracts files
            main_py_path = os.path.join(app_dir, '_internal', 'main.py')
    else:
        # Running as script
        app_dir = os.path.dirname(os.path.abspath(__file__))
        main_py_path = os.path.join(app_dir, 'main.py')

    print(f"App directory: {app_dir}")
    print(f"Main.py path: {main_py_path}")
    
    # Verify main.py exists
    if not os.path.exists(main_py_path):
        print(f"Error: main.py not found at {main_py_path}")
        if hasattr(sys, '_MEIPASS'):
            print(f"PyInstaller temp directory: {sys._MEIPASS}")
            print("Contents of temp directory:")
            if os.path.exists(sys._MEIPASS):
                for item in os.listdir(sys._MEIPASS):
                    print(f"  {item}")
        input("Press Enter to exit...")
        return

    # Change to the app directory
    os.chdir(app_dir)

    # Set up environment for Streamlit
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'False'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_UI_HIDE_TOP_BAR'] = 'true'
    os.environ['STREAMLIT_CLIENT_TOOLBAR_MODE'] = 'minimal'
    os.environ['STREAMLIT_THEME_BASE'] = 'light'
    os.environ['STREAMLIT_THEME_PRIMARY_COLOR'] = '#0066cc'
    os.environ['STREAMLIT_THEME_BACKGROUND_COLOR'] = '#ffffff'
    os.environ['STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR'] = '#f0f2f6'
    os.environ['STREAMLIT_THEME_TEXT_COLOR'] = '#262730'

    try:
        # Import streamlit and run directly using bootstrap
        print("Launching Streamlit app...")
        print("The app will be available at: http://localhost:8501")
        print("Use Ctrl+C to stop the application")

        import streamlit.web.bootstrap as bootstrap
        import streamlit.web.cli as stcli

        # Set up sys.argv to mimic command line arguments
        sys.argv = [
            'streamlit',
            'run',
            main_py_path,
            '--server.headless=true',
            '--server.port=8501',
            '--server.address=0.0.0.0',
            '--browser.gatherUsageStats=false',
            '--server.enableCORS=false',
            '--server.enableXsrfProtection=false',
            '--global.developmentMode=false',
            '--ui.hideTopBar=true',
            '--client.toolbarMode=minimal',
            '--theme.base=light',
            '--theme.primaryColor=#0066cc',
            '--theme.backgroundColor=#ffffff',
            '--theme.secondaryBackgroundColor=#f0f2f6',
            '--theme.textColor=#262730'
        ]

        # Use streamlit's CLI directly
        stcli.main()

    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error launching Streamlit: {e}")

        # Final fallback: try to import and run main.py directly
        try:
            print("Trying direct import of main.py...")
            import main
        except Exception as e2:
            print(f"Direct import failed: {e2}")
            print("Please check that all required dependencies are installed.")
            input("Press Enter to exit...")

if __name__ == "__main__":
    main()