from pythonforandroid.recipe import CythonRecipe
from os.path import join
import glob


class PyjniusRecipe(CythonRecipe):
    version = '1.6.1'
    url = 'https://github.com/kivy/pyjnius/archive/{version}.zip'
    name = 'pyjnius'
    depends = ['setuptools', 'six']
    site_packages_name = 'jnius'

    def prebuild_arch(self, arch):
        super().prebuild_arch(arch)
        self.patch_long_type(arch)

    def patch_long_type(self, arch):
        """Fix Python 2 'long' type usage in pyjnius for Python 3 compatibility"""
        build_dir = self.get_build_dir(arch.arch)
        pxi_files = glob.glob(join(build_dir, '**', '*.pxi'), recursive=True)
        pxi_files += glob.glob(join(build_dir, '**', '*.pyx'), recursive=True)
        for fpath in pxi_files:
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if 'long' not in content:
                    continue
                new_content = content
                replacements = [
                    ('isinstance(arg, long)', 'isinstance(arg, int)'),
                    ('isinstance(arg, (int, long))', 'isinstance(arg, int)'),
                    ('isinstance(py_arg, long)', 'isinstance(py_arg, int)'),
                    ('isinstance(py_arg, (int, long))', 'isinstance(py_arg, int)'),
                    (', long)', ', int)'),
                ]
                for old, new in replacements:
                    new_content = new_content.replace(old, new)
                if new_content != content:
                    with open(fpath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f'Patched pyjnius file: {fpath}')
            except Exception as e:
                print(f'Warning: could not patch {fpath}: {e}')


recipe = PyjniusRecipe()
