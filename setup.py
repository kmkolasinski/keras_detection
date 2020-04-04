from setuptools import setup

VERSION = '0.1'

setup(
    name='keras_detection',
    packages=['keras_detection'],
    version=VERSION,
    description='keras_detection',
    author='Krzysztof Kolasiński',
    author_email='kmkolasinski@gmail.com',
    url='https://github.com/kmkolasinski/keras_detection',
    keywords=['tensorflow', 'keras', 'object-detection'],
    classifiers=[],
    install_requires=[
        'numba==0.48.0',
        'numpy==1.16.4',
        'tsalib==0.2.1',
        'matplotlib>=3.1.*',
        'image-classifiers==1.0.0',
        'tqdm>=4.43.0',
        'tensorflow-addons>=0.8.*',
        'imgaug>=0.4',
        'cached_property>=1.5',
    ],
    include_package_data=True
)
