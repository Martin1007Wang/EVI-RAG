#!/usr/bin/env python

from pathlib import Path

from setuptools import find_packages, setup


def _load_requirements() -> list[str]:
    requirements_path = Path(__file__).resolve().parent / "requirements.txt"
    requirements: list[str] = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if requirement:
            requirements.append(requirement)
    return requirements


setup(
    name="evi-rag",
    version="0.0.1",
    description="EVI-RAG training and evaluation utilities",
    author="",
    author_email="",
    url="https://github.com/user/project",
    python_requires=">=3.8",
    install_requires=_load_requirements(),
    packages=find_packages(exclude=("tests", "tests.*")),
    py_modules=["text_encode_utils"],
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "evaluate_command = src.evaluate:main",
            "preprocess_command = src.preprocess:main",
        ]
    },
)
