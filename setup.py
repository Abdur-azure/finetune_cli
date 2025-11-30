"""
Setup script for LLM Fine-Tuning Framework.

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().splitlines()

setup(
    name="finetune-cli",
    version="2.0.0",
    description="Production-grade LLM fine-tuning framework with CLI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Abdur Rahman",
    author_email="",
    url="https://github.com/Abdur-azure/finetune_cli",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "mkdocs-minify-plugin>=0.7.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "finetune-cli=finetune_cli.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm fine-tuning lora qlora transformers machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/Abdur-azure/finetune_cli/issues",
        "Documentation": "https://Abdur-azure.github.io/finetune_cli",
        "Source": "https://github.com/Abdur-azure/finetune_cli",
    },
)