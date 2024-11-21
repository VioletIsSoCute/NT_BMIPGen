from setuptools import setup, find_packages

with open("README.md", "r",encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='NT_BMIPGen',
    version='0.4.4',
    author='Meng-Lin Tsai',
    author_email='mtsai47@wisc.edu',
    url = "https://avraamidougroup.che.wisc.edu",
    description='Help Generating nontrivial bilevel mixed integer problem',
    packages= find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.20.0",
        "pyomo>=6.0",
        "tqdm>=4.0.0",
        "matplotlib>=3.0.0",
        "pandas>=2.0.0"
    ],
    classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)