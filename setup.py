from setuptools import setup

setup(
    name='kwiklib',
    version='0.3.0',
    author='Klusta-Team',
    author_email='rossant@github',
    packages=[
              'kwiklib',
              'kwiklib.dataio',
              'kwiklib.dataio.tests',
              'kwiklib.scripts',
              'kwiklib.utils',
              'kwiklib.utils.tests',
              ],
    entry_points={
          'console_scripts':
              ['kwikkonvert = kwiklib.scripts.runkwikkonvert:main',
               ]},
    url='http://klusta-team.github.io',
    license='LICENSE.txt',
    description='Kwiklib, part of the KlustaSuite',
    # long_description=open('README.md').read(),
)
