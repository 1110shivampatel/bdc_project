"""Script to launch the Streamlit dashboard."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dashboard_path = project_root / "dashboards" / "streamlit_app.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Error: Streamlit is not installed.")
        print("Please install it with: pip install streamlit")
        sys.exit(1)
    
    print("Starting BDC Analytics Dashboard...")
    print(f"Dashboard will open at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print()
    
    # Use python -m streamlit instead of direct streamlit command
    # This works better on Windows and ensures we use the correct Python environment
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
        sys.exit(0)

