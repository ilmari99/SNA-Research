from setuptools import setup, find_packages

setup(
    name='RLFramework',
    version='1.0',
    packages=find_packages(),
    author='Ilmari Vahteristo',
    author_email='i.vahteristo@gmail.com',
    description='A simple reinforcement learning framework',
    url='https://github.com/ilmari99/RLFramework.git',
    install_requires=[
        "tensorflow==2.15",
        "matplotlib",
        "tqdm"
    ]
)