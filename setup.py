import os
from setuptools import setup, find_packages

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQ_MAIN = os.path.join(PROJECT_ROOT, "requirements.txt")
REQ_DEV = os.path.join(PROJECT_ROOT, "requirements-dev.txt")
REQ_DOCS = os.path.join(PROJECT_ROOT, "requirements-docs.txt")
README = os.path.join(PROJECT_ROOT, "README.md")

def _read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        
def _read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_requirements():
    return _read_lines(REQ_MAIN)

def get_requirements_dev():
    return _read_lines(REQ_DEV)

def get_requirements_docs():
    return _read_lines(REQ_DOCS)

def get_long_description():
    with open(README, "r", encoding="utf-8") as f:
        return f.read()

setup(
    name="swasd",
    version="0.1.0",
    description="Sliced-Wasserstein Automated Stationarity Detection (SWASD)",
    long_description=_read_text(README),
    long_description_content_type="text/markdown",
    author="Manushi Welandawe",
    author_email='manushiw@bu.edu',
    url="https://github.com/Manushi22/swasd",
    python_requires=">=3.10",

    package_dir={"": "swasd/src"},
    packages=find_packages(where="swasd/src"),

    include_package_data=True,
    install_requires=get_requirements(),
    extras_require={
        "dev": get_requirements_dev(),
        "docs": get_requirements_docs(),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    platforms="ALL",
)