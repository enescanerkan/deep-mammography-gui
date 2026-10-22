# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File for Mammogram Positioning Analysis

This spec file configures the build process for creating a standalone
Windows executable from the application.

Usage:
    pyinstaller mammogram_analysis.spec

Output:
    dist/MammogramAnalysis.exe
    (with MAMMOGRAM_ONEDIR=1: dist/MammogramAnalysis/ folder, MammogramAnalysis.exe + _internal)
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Project paths
project_root = Path('.').absolute()
src_path = project_root / 'src'

# One-folder build: starts faster (nothing to unpack) but has to be shipped as a folder
ONEDIR = os.environ.get('MAMMOGRAM_ONEDIR', '').strip().lower() in ('1', 'true', 'yes')

# Collect data files
datas_list = []
if (project_root / 'data').exists():
    datas_list.append((str(project_root / 'data'), 'data'))

# Include model weights in the executable
weights_path = project_root / 'weights'
if weights_path.exists():
    datas_list.append((str(weights_path), 'weights'))

# Window icon (assets/logo.png) and splash image
assets_path = project_root / 'assets'
if assets_path.exists():
    datas_list.append((str(assets_path), 'assets'))

# Analysis configuration
a = Analysis(
    [str(src_path / 'main.py')],
    pathex=[str(src_path)],
    binaries=[],
    datas=datas_list,
    hiddenimports=[
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'torch',
        'numpy',
        'cv2',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'pydicom',
        'skimage',
        'skimage.transform',
        'pandas',
        'scipy',
        'PIL',
        'requests',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(project_root / 'pyi_rth_0_torch_dll.py')],  # torch before Qt on Windows
    # Packages the application never imports; each one only adds unpacking time at start-up
    excludes=[
        'tkinter',
        'pytest',
        'jupyter',
        'notebook',
        'IPython',
        'torchvision',
        'sklearn',
        'tensorboard',
        'torch.utils.tensorboard',
        'PyQt5.QtWebEngine',
        'PyQt5.QtWebEngineCore',
        'PyQt5.QtWebEngineWidgets',
        'PyQt5.QtQml',
        'PyQt5.QtQuick',
        'PyQt5.QtMultimedia',
        'PyQt5.QtBluetooth',
        'PyQt5.QtSql',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)



# Create PYZ archive
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Splash screen shown by the bootloader while the executable unpacks; main.py closes it
# once the window is up. It needs Tcl/Tk in the build Python, so the build goes on
# without it if Tcl/Tk is missing.
splash_parts = []
splash_image = project_root / 'assets' / 'splash.png'
if splash_image.exists():
    try:
        splash = Splash(
            str(splash_image),
            binaries=a.binaries,
            datas=a.datas,
            text_pos=(40, 268),
            text_size=9,
            text_color='#98a3b0',
            minify_script=True,
            always_on_top=True,
        )
        splash_parts = [splash, splash.binaries]
    except (Exception, SystemExit) as exc:  # PyInstaller exits instead of raising when Tcl/Tk is missing
        print(f"WARNING: building without a splash screen ({exc!r})")

# Create executable
exe_options = dict(
    name='MammogramAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-packed DLLs have to be unpacked again at every start
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / 'assets' / 'icon.ico'),
)

if ONEDIR:
    exe = EXE(pyz, a.scripts, *splash_parts[:1], [], exclude_binaries=True, **exe_options)
    coll = COLLECT(
        exe,
        *splash_parts[1:],
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name='MammogramAnalysis',
    )
else:
    exe = EXE(pyz, a.scripts, *splash_parts, a.binaries, a.zipfiles, a.datas, [], **exe_options)
