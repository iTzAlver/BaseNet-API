# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

REQUIREMENTS = ['tensorflow>=2.10.0',
                # 'tensorflow-gpu==2.10.*',
                'graphviz >= 0.20.1',
                'matplotlib>=3.6.1',
                'future>=0.18.2',
                'Pillow>=9.2.0',
                'numpy>=1.23.4',
                'ttkwidgets>=0.12.1',
                'pydot>=1.4.2',
                'keras>=2.9.0',
                'setuptools>=60.2.0',
                'PyYAML>=6.0',
                'tensorboard>=2.9.1',
                'psutil>=5.9.3',
                'pynvml>=11.4.1']


setuptools.setup(
    name='basenet_api',
    version='1.5.2',
    author='Palomo-Alonso, Alberto',
    author_email='a.palomo@edu.uah',
    description='Basenet API: A simpler way to build ML models.',
    keywords='deeplearning, ml, api',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/iTzAlver/basenet_api.git',
    project_urls={
        'Documentation': 'https://htmlpreview.github.io/?https://github.com/iTzAlver/basenet_api/blob/'
                         'main/doc/basenet.html',
        'Bug Reports': 'https://github.com/iTzAlver/basenet_api/issues',
        'Source Code': 'https://github.com/iTzAlver/basenet_api.git',
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
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License'
    ],
    python_requires='>=3.6',
    # install_requires=['Pillow'],
    extras_require={
        'dev': ['check-manifest'],
    },
    include_package_data=True,
    install_requires=REQUIREMENTS
)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
