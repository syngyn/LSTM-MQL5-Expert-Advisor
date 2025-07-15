import os
import sys
import time
import subprocess
import signal

# --- Configuration ---
TERMINAL_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe" 
CONFIG_FILE_PATH = "C:\\Users\\jason\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\config\\terminal.ini"
EXPORT_SCRIPT_NAME = "ExportHistory.mq5"

# The directory where the Python scripts are located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PID_FILE = os.path.join(SCRIPT_DIR, "daemon.pid")

# --- Functions ---

def print_header(message):
    print(f"\n{'='*60}\n=== {message.upper()} ===\n{'='*60}")

def stop_daemon():
    # This function is correct.
    print_header("Step 1: Stopping existing daemon")
    if not os.path.exists(PID_FILE):
        print("No PID file found. Daemon is likely not running. Continuing...")
        return
    try:
        with open(PID_FILE, 'r') as f: pid = int(f.read().strip())
        print(f"Found daemon process with PID: {pid}. Attempting to stop...")
        os.kill(pid, signal.SIGTERM)
        time.sleep(5)
        if not os.path.exists(PID_FILE): print("Daemon shut down successfully.")
        else: print("Warning: Forcing cleanup of stale PID file."); os.remove(PID_FILE)
    except (ProcessLookupError, PermissionError):
        print(f"Process with PID {pid} not found. Cleaning up stale PID file.")
        if os.path.exists(PID_FILE): os.remove(PID_FILE)
    except Exception as e:
        print(f"An error occurred while stopping the daemon: {e}"); sys.exit(1)


def export_new_data():
    print_header("Step 2: Exporting new data from MetaTrader 5")
    if not os.path.exists(TERMINAL_PATH):
        print(f"ERROR: MT5 not found at '{TERMINAL_PATH}'"); sys.exit(1)
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"ERROR: MT5 config file not found at '{CONFIG_FILE_PATH}'"); sys.exit(1)
    
    # --- FIX: The MQL5 script name is relative to the terminal's context, not our python script folder.
    # The terminal knows where its /MQL5/Scripts/ folder is. We only need to provide the name.
    # The check for its existence is removed as it was causing the pathing confusion.
    # The subprocess call will fail if the script is not found by the terminal itself.
    
    print(f"Starting MetaTrader 5 using profile config: {CONFIG_FILE_PATH}")
    print(f"Asking terminal to run script: {EXPORT_SCRIPT_NAME}")
    
    # The terminal resolves "ExportHistory.mq5" to its own /MQL5/Scripts/ folder.
    command_args = [TERMINAL_PATH, f"/config:{CONFIG_FILE_PATH}", f"/script:{EXPORT_SCRIPT_NAME}"]
    
    try:
        # We must set the working directory for MT5 to its "Files" folder so it knows where to save the CSVs.
        # Your python scripts are in a subfolder of "Files", so SCRIPT_DIR is the correct place to save.
        subprocess.run(command_args, timeout=300, check=True, cwd=SCRIPT_DIR)
        print("Data export script completed successfully.")
    except Exception as e:
        print(f"ERROR: Data export process failed. Error: {e}"); sys.exit(1)


def train_models():
    # This function is now correct.
    print_header("Step 3: Training new models")
    try:
        classification_script_path = os.path.join(SCRIPT_DIR, "train_LSTM.py")
        regression_script_path = os.path.join(SCRIPT_DIR, "train_regression.py")

        print(f"\n--- Training Classification Model ---")
        print(f"Executing: \"{sys.executable}\" \"{classification_script_path}\"")
        subprocess.run([sys.executable, classification_script_path], check=True, cwd=SCRIPT_DIR)
        
        print(f"\n--- Training Regression Model ---")
        print(f"Executing: \"{sys.executable}\" \"{regression_script_path}\"")
        subprocess.run([sys.executable, regression_script_path], check=True, cwd=SCRIPT_DIR)
        
        print("Model training complete.")
    except Exception as e:
        print(f"ERROR: Model training failed: {e}"); sys.exit(1)

def start_daemon():
    """Starts the daemon.py script in a new background process using an absolute path."""
    print_header("Step 4: Starting new daemon")
    try:
        # --- FIX: Ensure the filename matches your actual file exactly ---
        # It's likely named "daemon.py" (lowercase 'd'), not "Daemon.py".
        daemon_script_path = os.path.join(SCRIPT_DIR, "daemon.py") # <-- CORRECTED FILENAME

        print(f"Executing in background: \"{sys.executable}\" \"{daemon_script_path}\"")
        if not os.path.exists(daemon_script_path):
             print(f"FATAL ERROR: The daemon script was not found at the expected path: {daemon_script_path}")
             sys.exit(1)
             
        subprocess.Popen([sys.executable, daemon_script_path], cwd=SCRIPT_DIR)
        print("Daemon process started successfully.")
    except Exception as e:
        print(f"ERROR: Failed to start the daemon process. Error: {e}")
        sys.exit(1)

# --- Main Execution Block ---
if __name__ == "__main__":
    stop_daemon()
    export_new_data()
    train_models()
    start_daemon()
    print(f"\n{'*'*60}\n=== AUTOMATED RETRAINING PROCESS COMPLETE ===\n{'*'*60}")