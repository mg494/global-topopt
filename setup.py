from setuptools import setup

def read(filename):
    with open(filename) as fp:
        text = fp.read()
    return text

setup(
    name='topopt',
    version="0.1.0",
    author='Marc Gehring',
    author_email='marc.gehring@web.de',
    url='https://github.com/mg494',
    description='Topology Optimiziation in Python',
    long_description=read('README.md'),
    license='GPL-3.0 License',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
    ],
    package_dir={"topopt":"topopt","fem":"fem","utils":"utils"}
)