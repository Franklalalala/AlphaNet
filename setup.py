from setuptools import setup, find_packages

setup(
    name='alphanet',
    version="0.0.1",
    packages=find_packages(include=['alphanet', 'alphanet.*']),
    install_requires=[
        'pyfiglet',
        'rich',
        'ase',
        'rdkit',
        'pydantic',
        'scikit-learn'
        
    ],
    entry_points={
        "console_scripts": [
            "alpha-train=alphanet.cli:main",
            "alpha-eval=alphanet.eval:evaluate",
            "alpha-conv=alphanet.state_dict:cli",
            "alpha-freeze=alphanet.freeze:freeze_model"
        ]
    },
    python_requires='>=3.8'
)
