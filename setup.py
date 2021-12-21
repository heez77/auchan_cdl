from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'albumentations==1.1.0',
    'effdet==0.2.4',
    'ensemble_boxes==1.0.7',
    'fast_bert==1.9.9',
    'fastcore==1.3.27',
    'matplotlib==3.5.0',
    'nltk==3.6.5',
    'numpy==1.20.0',
    'pandas==1.3.4',
    'Pillow==8.4.0',
    'pytorch_lightning==1.3.5',
    'scikit_learn==1.0.1',
    'tensorflow==2.7.0',
    'Unidecode==1.3.2'
]

setup(
    name = 'trainer',
    version = '0.7',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True
)