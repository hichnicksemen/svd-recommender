from setuptools import setup, find_packages

setup(
    name='svd-recommender',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn'
    ],
    author='Semen Lobachevskiy',
    author_email='hichnick@gmail.com',
    description='A simple collaborative recommender system using SVD',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/hichnicksemen/svd-recommender',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
