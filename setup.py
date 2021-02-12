 #!/usr/bin/env python3
import os
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.egg_info import egg_info
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.test import test as TestCommand
from shutil import copyfile, copymode

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        # update the submodule
        print("Updating git submodules...")
        import git
        repo = git.Repo(os.path.dirname(os.path.realpath(__file__)))
        repo.git.submodule('update', '--init', '--recursive')
        
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")
            
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)


def make_git_file():
    print("Creating pygama/git.py")
    try:
        import git
        repo = git.Repo(os.path.dirname(os.path.realpath(__file__)))
        with open(repo.working_tree_dir + '/pygama/git.py', 'w') as f:
            f.write("branch = '" + repo.git.describe('--all') + "'\n")
            f.write("revision = '" + repo.head.commit.hexsha +"'\n")
            f.write("commit_date = '" + str(repo.head.commit.committed_datetime) + "'\n")
    except Exception as ex:
        print(ex)
        print('continuing...')

#Add a git hook to clean jupyter notebooks before commiting
def clean_jupyter_notebooks():
    import git
    repo = git.Repo(os.path.dirname(os.path.realpath(__file__)))
    with repo.config_writer('repository') as config:
        try:
            import nbconvert
            if nbconvert.__version__[0] < '6': #clear output
                fil=""" "jupyter nbconvert --stdin --stdout --log-level=ERROR\\
                --to notebook --ClearOutputPreprocessor.enabled=True" """
            else: # also clear metadata
                fil=""" "jupyter nbconvert --stdin --stdout --log-level=ERROR\\
                --to notebook --ClearOutputPreprocessor.enabled=True\\
                --ClearMetadataPreprocessor.enabled=True" """                
        except:
            # if nbconvert (part of jupyter) is not installed, disable filter
            fil = "cat"

        config.set_value('filter "jupyter_clear_output"', 'clean', fil)
        config.set_value('filter "jupyter_clear_output"', 'smudge', 'cat')
        config.set_value('filter "jupyter_clear_output"', 'required', 'false')
        

# run during installation; this is when files get copied to build dir
class PygamaBuild(build_py):
    def run(self):
        make_git_file()
        clean_jupyter_notebooks()
        build_py.run(self)

# run during local installation; in this case build_py isn't run...
class PygamaDev(develop):
    def run(self):
        make_git_file()
        clean_jupyter_notebooks()
        develop.run(self)

setup(
    name='pygama',
    version='0.5',
    author='LEGEND',
    author_email='wisecg@uw.edu',
    description='Python package for decoding and processing digitizer data',
    long_description='',
    packages=find_packages(),
    install_requires=[
        'scimath',
        'numba',
        'parse',
        'GitPython',
        'tinydb',
        'pyFFTW',
        'h5py',
        'numpy',
        'pandas',
        'matplotlib'
        # 'fcutils @ https://github.com/legend-exp/pyfcutils.git#egg=1.0.0'
    ],
    cmdclass=dict(build_ext=CMakeBuild, build_py=PygamaBuild, develop=PygamaDev),
    zip_safe=False,
)
