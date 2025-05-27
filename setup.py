from setuptools import setup, find_packages

setup(
    name="repo_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gitpython>=3.1.30",
        "pandas>=1.3.5",
        "lizard>=1.17.10",
        "pylint>=2.15.0",
        "requests>=2.28.1",
        "tqdm>=4.64.0",
        "psutil>=5.9.0",
    ],
    entry_points={
        'console_scripts': [
            'repo-analyzer=repo_analyzer.main:main',
        ],
    },
    author="Simone Silvestri",
    author_email="s.silvestri15@studenti.unisa.it",
    description="Strumento per analizzare metriche di qualità del codice in repository Git Python",
    keywords="git, code quality, metrics, python",
    python_requires='>=3.6',
)