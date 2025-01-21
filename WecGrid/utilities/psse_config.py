import os
import sys

def configure_psse_paths():
    """
    Adds PSSE paths to the system PATH and Python sys.path for runtime use.
    """
    psse_path = r"C:\Program Files\PTI\PSSE35\35.3"  # Update as needed
    additional_paths = [
        os.path.join(psse_path, subdir)
        for subdir in ["PSSPY37", "PSSBIN", "PSSLIB", "EXAMPLE"]
    ]

    for path in additional_paths:
        if path not in sys.path:
            sys.path.append(path)
    os.environ["PATH"] += ";" + ";".join(additional_paths)

    try:
        import psse35
        import psspy
        psse35.set_minor(3)
        psspy.psseinit(50)
        print("PSSE paths successfully configured.")
    except ImportError as e:
        raise ImportError(f"Failed to configure PSSE paths: {e}")