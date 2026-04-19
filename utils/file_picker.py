import platform
import subprocess


def pick_directory() -> str | None:
    """
    Open the OS-native directory picker and return the selected absolute path.
    Returns None if the user cancels or the dialog cannot be launched.

    Implemented via subprocess so it works from Streamlit's script-runner
    thread (tkinter requires the process main thread and breaks on macOS).
    """
    system = platform.system()
    try:
        if system == "Darwin":
            script = (
                'tell application "System Events" to activate\n'
                'POSIX path of (choose folder with prompt "Select staging directory")'
            )
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                return None
            path = result.stdout.strip().rstrip("/")
            return path or None

        if system == "Windows":
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms | Out-Null; "
                "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
                "$f.Description = 'Select staging directory'; "
                "if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
                "{ Write-Output $f.SelectedPath }"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                return None
            return result.stdout.strip() or None

        # Linux / other: try zenity, then kdialog
        for cmd in (
            ["zenity", "--file-selection", "--directory", "--title=Select staging directory"],
            ["kdialog", "--getexistingdirectory"],
        ):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    return result.stdout.strip() or None
            except FileNotFoundError:
                continue
        return None

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
