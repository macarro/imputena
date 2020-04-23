import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='imputena',
    version='0.2',
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
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Financial and Insurance Industry'
    ],
    python_requires='>=3.5',
    install_requires=[
        'pandas',
        'numpy',
        'statsmodels',
        'sklearn'
    ],
)
