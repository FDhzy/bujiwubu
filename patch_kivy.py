"""
Patch Kivy Cython files for compatibility with newer Cython/Python3.
Fixes:
1. kivy/weakproxy.pyx: remove Python2 'long' type usage
2. kivy/lib/gstplayer/_gstplayer.pyx: add 'noexcept' to callback types
"""
import glob
import os

# ─── Patch 1: weakproxy.pyx ── 'long' is not a type in Python3/Cython3 ───────

weakproxy_patterns = [
    os.path.expanduser('~/.buildozer/**/kivy/weakproxy.pyx'),
    '.buildozer/**/kivy/weakproxy.pyx',
]

for pat in weakproxy_patterns:
    for f in glob.glob(pat, recursive=True):
        try:
            content = open(f, encoding='utf-8', errors='ignore').read()
            new = content
            # Remove the entire __long__ method (Python2 artifact)
            import re
            new = re.sub(
                r'    def __long__\(self\):\s*\n        return long\(self\.__ref__\(\)\)\s*\n',
                '',
                new
            )
            # Also handle plain 'long(' replacement as fallback
            new = new.replace('return long(', 'return int(')
            new = new.replace('(int, long)', 'int')
            new = new.replace('isinstance(arg, long)', 'isinstance(arg, int)')
            if new != content:
                open(f, 'w', encoding='utf-8').write(new)
                print('Patched weakproxy.pyx:', f)
        except Exception as e:
            print('Skip weakproxy.pyx:', f, e)


# ─── Patch 2: _gstplayer.pyx ── callback noexcept incompatibility ────────────

gstplayer_patterns = [
    os.path.expanduser('~/.buildozer/**/kivy/lib/gstplayer/_gstplayer.pyx'),
    '.buildozer/**/kivy/lib/gstplayer/_gstplayer.pyx',
]

for pat in gstplayer_patterns:
    for f in glob.glob(pat, recursive=True):
        try:
            content = open(f, encoding='utf-8', errors='ignore').read()
            new = content
            # Add 'noexcept' to the two callback function definitions
            # _on_appsink_sample
            new = re.sub(
                r'(cdef void _on_appsink_sample\([^)]+\))\s*(?!noexcept)(\s*(?:with|nogil|:))',
                r'\1 noexcept\2',
                new
            )
            # _on_gstplayer_message
            new = re.sub(
                r'(cdef void _on_gstplayer_message\([^)]+\))\s*(?!noexcept)(\s*(?:with|nogil|:))',
                r'\1 noexcept\2',
                new
            )
            # Also patch the type declarations in the struct/typedef
            new = new.replace(
                'except * nogil',
                'noexcept nogil'
            )
            if new != content:
                open(f, 'w', encoding='utf-8').write(new)
                print('Patched _gstplayer.pyx:', f)
            else:
                print('No change needed for _gstplayer.pyx:', f)
        except Exception as e:
            print('Skip _gstplayer.pyx:', f, e)

# ─── Patch 3: lib2to3 Python2 test data files ───────────────────────────────
# These files contain Python 2 syntax (print "...") and crash Python 3 compiler

import os

lib2to3_data_patterns = [
    os.path.expanduser('~/.buildozer/**/lib2to3/tests/data'),
    '.buildozer/**/lib2to3/tests/data',
]

py2_files_removed = 0
for pat in lib2to3_data_patterns:
    for data_dir in glob.glob(pat, recursive=True):
        if os.path.isdir(data_dir):
            for fname in os.listdir(data_dir):
                if fname.endswith('.py'):
                    fpath = os.path.join(data_dir, fname)
                    try:
                        # Try to compile; if it fails it's Python2 syntax
                        with open(fpath, encoding='utf-8', errors='ignore') as f:
                            src = f.read()
                        try:
                            compile(src, fpath, 'exec')
                        except SyntaxError:
                            # Rename to .py2 so buildozer ignores it
                            os.rename(fpath, fpath + '2')
                            print('Disabled py2 file:', fpath)
                            py2_files_removed += 1
                    except Exception as e:
                        print('Skip:', fpath, e)

print(f'Disabled {py2_files_removed} Python2 test files.')
print('patch_kivy.py done.')
