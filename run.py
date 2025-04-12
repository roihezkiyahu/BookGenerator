import os
import subprocess
import sys
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

def run_streamlit_app(args: Optional[List[str]] = None) -> None:
    """
    Run the Streamlit application.
    
    Args:
        args: Additional arguments to pass to streamlit run
    """
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    
    if args:
        cmd.extend(args)
    
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)
    except Exception as e:
        logger.error(f"Error running Streamlit app: {str(e)}")


if __name__ == "__main__":
    run_streamlit_app()    