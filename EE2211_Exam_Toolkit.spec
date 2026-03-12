# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['EE2211_GUI.py'],
    pathex=[],
    binaries=[],
    datas=[('C:/Users/llk40/AppData/Local/Programs/Python/Python311/Lib/site-packages/matplotlib/mpl-data', 'matplotlib/mpl-data')],
    hiddenimports=['numpy', 'matplotlib', 'sympy', 'sympy.parsing.sympy_parser', 'scipy', 'scipy.optimize'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pandas', 'sklearn', 'torch', 'torchvision', 'torchaudio', 'tensorflow', 'IPython', 'notebook', 'pytest'],
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
    name='EE2211_Exam_Toolkit',
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
