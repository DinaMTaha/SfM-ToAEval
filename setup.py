import re
from setuptools import find_packages, setup

PACKAGE_NAME = 'sfm_toaeval'
SOURCE_DIRECTORY = 'src'
SOURCE_PACKAGE_REGEX = re.compile(rf'^{SOURCE_DIRECTORY}')

source_packages = find_packages(include=[SOURCE_DIRECTORY, f'{SOURCE_DIRECTORY}.*'])
proj_packages = [SOURCE_PACKAGE_REGEX.sub(PACKAGE_NAME, name) for name in source_packages]

setup(
    name=PACKAGE_NAME,
    packages=proj_packages,
    package_dir={PACKAGE_NAME: SOURCE_DIRECTORY},
    version='0.1.0',
    description='Trade-off Aware Evaluation of Feature Extraction Algorithms in Structure from Motion',
    author='Dina M. Taha',
    author_email='dina.taha@feng.bu.edu.eg',
    license='MIT',
)
