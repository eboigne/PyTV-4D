from setuptools import setup

requirements = [
	'numpy',
    'matplotlib',
	'pytorch>=1.5.0'
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
    include_package_data=True, # With this, the non .py files specified in MANIFEST.in are included
    package_data={'pytv': ['pytv/media', 'pytv/media/*']},
    keywords='pytv',
)
