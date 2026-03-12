import glob
import os

patterns = [
    os.path.expanduser('~/.buildozer/**/*.pxi'),
    os.path.expanduser('~/.buildozer/**/*.pyx'),
    '.buildozer/**/*.pxi',
    '.buildozer/**/*.pyx',
]

patched = 0
for pat in patterns:
    for f in glob.glob(pat, recursive=True):
        try:
            content = open(f, encoding='utf-8', errors='ignore').read()
            if 'long' not in content:
                continue
            new = content
            new = new.replace('return long(', 'return int(')
            new = new.replace('(int, long)', 'int')
            new = new.replace('isinstance(arg, long)', 'isinstance(arg, int)')
            new = new.replace('isinstance(py_arg, long)', 'isinstance(py_arg, int)')
            new = new.replace('(long)', '(int)')
            if new != content:
                open(f, 'w', encoding='utf-8').write(new)
                print('Patched:', f)
                patched += 1
        except Exception as e:
            print('Skip:', f, e)

print('Total patched:', patched)
