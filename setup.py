import setuptools
import os 
import genomix


# Get the long description from the README file
readme_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')
with open(readme_file) as f:
    long_description = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name = genomix.__name__,
        version = genomix.__version__,
        author = genomix.__author__,
        author_email = genomix.__email__,
        project_urls = {'Source': 'https://github.com/gvishwak/genomix', },
        description = 'A python implementation of genetic algorithm for solving real-valued optimization problems.',
        long_description = long_description,
        long_description_content_type = "text/markdown",
        keywords = ['Genetic Algorithm', ],
        license = 'BSD-3C',
        packages = setuptools.find_packages(),
        include_package_data = True,

        install_requires = ['numpy', 'pandas', 'seaborn', 'matplotlib'],
        extras_require = {
            'docs': [
                'sphinx',
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
                'nbsphinx'
            ],
            'tests': [
                'pytest',
                'pytest-pep8',
                'tox'
            ],
        },
        tests_require = [
            'pytest',
            'pytest-pep8',
            'tox',
        ],
        zip_safe = False,
    )
