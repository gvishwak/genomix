import setuptools
import os
import genomix

from genomix.__version__ import __version__


# Get the long description from the README file
readme_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')
with open(readme_file) as f:
    long_description = f.read()


if __name__ == "__main__":
    setuptools.setup(
        name = genomix.__name__,
        version = __version__,
        author = 'Gaurav Vishwakarma',
        project_urls = {'Source': 'https://github.com/gvishwak/genomix', },
        description = 'A Python implementation of real-valued genetic algorithms for hyperparameter optimization in machine learning, particularly focused on applications in chemistry and materials science.',
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
