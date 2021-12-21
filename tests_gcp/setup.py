from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'numpy==1.20.0'
]

setup(
    name='Gertrude',
    version='0.1',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True
)