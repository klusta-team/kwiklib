from setuptools import setup

setup(
    name='kwiklib',
    version='0.3.7',
    author='KwikTeam',
    author_email='rossant@github',
    packages=[
              'kwiklib',
              'kwiklib.dataio',
              'kwiklib.dataio.tests',
              'kwiklib.scripts',
              'kwiklib.utils',
              'kwiklib.utils.tests',
              ],
    # entry_points={
          # 'console_scripts':
              # ['kwikkonvert = kwiklib.scripts.runkwikkonvert:main',
               # ]},
    url='http://klusta-team.github.io',
    license='LICENSE.txt',
    description='Kwiklib (legacy)',
    # long_description=open('README.md').read(),
)
