# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('app.py', 'source'), ('requirements.txt', 'source'), ('requirements-build.txt', 'source'), ('README.md', 'source'), ('run_app.bat', 'source'), ('build_exe.bat', 'source'), ('arduino_morse_leds.ino', 'source'), ('siscom', 'source/siscom')],
    hiddenimports=['matplotlib.backends.backend_tkagg'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'torch', 'torchmetrics', 'IPython', 'jupyter', 'pyarrow', 'numba', 'llvmlite', 'openpyxl', 'lxml', 'cryptography', 'fsspec'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SisCom Morse',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
