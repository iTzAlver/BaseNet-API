# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import setuptools
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='basenet',
    version='1.0.0',
    author='Palomo-Alonso, Alberto',
    author_email='a.palomo@edu.uah',
    description='Basenet API: A simpler way to build ML models.',
    keywords='deeplearning, ml, api',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/iTzAlver/cnet.git',
    project_urls={
        'Documentation': 'https://htmlpreview.github.io/?https://github.com/iTzAlver/basenet_api/blob/'
                         'main/doc/basenet.html',
        'Bug Reports': 'https://github.com/iTzAlver/cnet/issues',
        'Source Code': 'https://github.com/iTzAlver/cnet.git',
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        # see https://pypi.org/classifiers/
        'Development Status :: 5 - Production/Stable',

        'Topic :: Software Development :: Build Tools',

        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache License'
    ],
    python_requires='>=3.6',
    # install_requires=['Pillow'],
    extras_require={
        'dev': ['check-manifest'],
    },
)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
