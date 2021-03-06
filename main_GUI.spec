# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['main_GUI.py'],
             pathex=[],
             binaries=[],
             datas=[('button/add.png','button'),
		   ('button/edit.png','button'),
		   ('button/forgetting.png','button'),
		   ('button/learning.png','button'),
		   ('button/memorized.png','button'),
		   ('button/add_hov.png','button'),
		   ('button/add_pressed.png','button'),
		   ('button/edit_hov.png','button'),
		   ('button/edit_pressed.png','button'),
		   ('button/forgetting_hov.png','button'),
		   ('button/forgetting_pressed.png','button'),
		   ('button/learning_hov.png','button'),
		   ('button/learning_pressed.png','button'),
		   ('button/mem_hov.png','button'),
		   ('button/mem_pressed.png','button'),
		   ('Files/class_map.csv', '.'),
		   ('Files/crypts.csv', '.'),
		   ('Files/data_map.csv', '.'),
		   ('Files/focus.csv', '.'),
		   ('Files/mach_data_map.csv', '.'),
		   ('Files/settings.csv', '.'),
		   ('Files/time_lapse.csv', '.')
	     ],
             hiddenimports=['sklearn.utils._typedefs'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [('W ignore', None, 'OPTION')],
          exclude_binaries=True,
          name='main_GUI',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Study_Help')

app = BUNDLE(coll,
             name='LFM.app',
             icon='button/mister.icns',
             bundle_identifier=None)
