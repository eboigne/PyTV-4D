from setuptools import setup

requirements = [
	'numpy',
	'pytorch'
]

setup(
    name='PyTV',
    version='0.1.0',
    description='Short description',
    license='GNUv3',
    author='Emeric Boigne',
    author_email='emericboigne@gmail.com',
    url='https://github.com/eboigne/PyTV',
    packages=['pytv'],
    install_requires=requirements,
    keywords='pytv',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
