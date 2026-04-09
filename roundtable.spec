# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Roundtable

import os

block_cipher = None

# Get the directory where this spec file is
spec_dir = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    ['launcher.py'],
    pathex=[spec_dir],
    binaries=[],
    datas=[
        ('templates', 'templates'),  # Include templates folder
        ('sd-rpg', 'sd-rpg'),        # Include sd-rpg for image generation
    ],
    hiddenimports=[
        'web_app',
        'config',
        'providers',
        'image_gen',
        'memory',
        'dm_agents',
        'anthropic',
        'openai',
        'httpx',
        'pydantic',
        'pydantic_settings',
        'flask',
        'flask_cors',
        'jinja2',
        'PIL',
        'PIL.Image',
        'requests',
        'websocket',
        'dotenv',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Roundtable',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console visible so user knows it's running
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Untitled-1.ico',
)
