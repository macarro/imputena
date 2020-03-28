import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='imputena',
    version='0.0.1',
    description='Package that allows both automated and customized treatment '
            'of missing values in datasets using Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url='http://github.com/macarro/imputena',
    author='Miguel Macarro',
    author_email='migmackle@alum.us.es',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas',
        'numpy'
    ],
)
