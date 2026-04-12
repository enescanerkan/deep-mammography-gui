# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File for Mammogram Positioning Analysis

This spec file configures the build process for creating a standalone
Windows executable from the application.

Usage:
    pyinstaller mammogram_analysis.spec

Output:
    dist/MammogramAnalysis.exe
"""

import sys
from pathlib import Path

block_cipher = None

# Project paths
project_root = Path('.').absolute()
src_path = project_root / 'src'

# Collect data files
datas_list = []
if (project_root / 'data').exists():
    datas_list.append((str(project_root / 'data'), 'data'))

# Include model weights in the executable
weights_path = project_root / 'weights'
if weights_path.exists():
    datas_list.append((str(weights_path), 'weights'))

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
        'torchvision',
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
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'pytest',
        'jupyter',
        'notebook',
        'IPython',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)



# Create PYZ archive
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MammogramAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one: 'assets/icon.ico'
)
