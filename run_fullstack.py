import subprocess
import sys
import time
import os
import urllib.error
import urllib.request
import webbrowser


APP_URL = "http://127.0.0.1:5000"
READY_URL = f"{APP_URL}/"


def wait_for_server(timeout_seconds: int = 30) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(READY_URL, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            time.sleep(0.5)
    return False


def main() -> int:
    popen_kwargs = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen([sys.executable, "app.py"], **popen_kwargs)

    try:
        if wait_for_server():
            webbrowser.open(APP_URL)
            print(f"Frontend is available at {APP_URL}")
        else:
            print("Server did not become ready in time. Check app logs in this terminal.")

        return process.wait()
    except KeyboardInterrupt:
        terminate_on_interrupt = os.getenv("FULLSTACK_STOP_ON_INTERRUPT", "0") == "1"
        if terminate_on_interrupt:
            print("\nStopping application...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            return 0

        print("\nLauncher interrupted, Flask server left running.")
        print("Set FULLSTACK_STOP_ON_INTERRUPT=1 to stop Flask when this launcher is interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
 