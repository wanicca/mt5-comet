from setuptools import setup, find_packages

setup(
    name='MyPackageName',
    version='1.0.0',
    url='https://github.com/mypackage.git',
    author='Author Name',
    author_email='author@gmail.com',
    description='Description of my package',
    #packages=find_packages(),    
    packages= ['rainbow','rainbow.preparation'],
    #package_dir={'': 'src'},
    #install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)
