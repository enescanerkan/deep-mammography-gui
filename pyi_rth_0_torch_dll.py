# PyInstaller runtime hook, frozen Windows only.
# The name pyi_rth_0_* sorts before the built-in pyi_rth_pyqt5, so torch is initialized
# before that hook puts PyQt5's Qt5\bin on the DLL search path: PyQt5 ships an older MSVC
# runtime there, and c10.dll fails with WinError 1114 when it picks that one up.
import os
import sys

if sys.platform == "win32" and getattr(sys, "frozen", False):
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(sys.argv[0]))

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # torch\lib first, then the other native-library folders of the bundle
    dll_dirs = []
    for sub in ("torch\\lib", "torch", "numpy.libs", "scipy.libs", "pandas.libs", "cv2"):
        path = os.path.join(base, sub)
        if os.path.isdir(path):
            dll_dirs.append(path)
    dll_dirs.append(base)
    for path in dll_dirs:
        try:
            os.add_dll_directory(path)
        except OSError:
            pass
    os.environ["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + os.environ.get("PATH", "")

    try:
        import torch  # noqa: F401
    except Exception:
        import traceback

        # With console=False the traceback would be lost; keep it next to the executable
        log_path = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "torch_import_error.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        except OSError:
            pass
        raise
