from setuptools import setup, find_packages


setup(
    name='poseval',
    version='0.1.0',
    packages=find_packages(),
    description='poseval',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',

    install_requires=[
        'click',
        'motmetrics>=1.2',
        'shapely',
        'tqdm',
    ],
    extras_require={
        'dev': [
            'pylint',
            'pytest',
        ],
    },
)
