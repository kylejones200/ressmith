"""
Setup configuration for PetroSmith package
"""

from setuptools import setup, find_packages
import os

# Read the long description from README
def read_long_description():
    """Read README for long description"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

def _read_pyproject():
    """Read pyproject.toml for single source of truth (version, dependencies)."""
    import tomli
    pyproject_path = os.path.join(os.path.dirname(__file__), 'pyproject.toml')
    if os.path.exists(pyproject_path):
        with open(pyproject_path, 'rb') as f:
            return tomli.load(f)
    return {}

def get_version():
    """Read version from pyproject.toml."""
    p = _read_pyproject()
    return p.get('project', {}).get('version', '0.3.0')

def get_install_requires():
    """Read install_requires from pyproject.toml [project.dependencies]."""
    p = _read_pyproject()
    return p.get('project', {}).get('dependencies', [])

def get_extras_require():
    """Read extras_require from pyproject.toml [project.optional-dependencies]."""
    p = _read_pyproject()
    return p.get('project', {}).get('optional-dependencies', {})

setup(
    name='petrosmith',
    version=get_version(),
    author='PetroSmith Team',
    author_email='info@petrosmith.dev',
    description='Comprehensive Petroleum Engineering Library',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/petrosmith',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
    entry_points={
        'console_scripts': [
            'petrosmith=petrosmith.cli.main:cli',
        ],
    },
    keywords='petroleum engineering drilling reservoir well-testing geomechanics',
    project_urls={
        'Documentation': 'https://github.com/yourusername/petrosmith#readme',
        'Source': 'https://github.com/yourusername/petrosmith',
        'Tracker': 'https://github.com/yourusername/petrosmith/issues',
    },
)
