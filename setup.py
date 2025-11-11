from setuptools import find_packages, setup

setup(
    name="axiom",
    version="0.1.0",
    author="Ishan Singh",
    author_email="ishans2404@gmail.com",
    description="Modular research library for Efficient Axial Networks and future architectures",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "tf": ["tensorflow>=2.13.0", "keras>=2.13.0"],
        "torch": ["torch>=2.1.0", "torchvision>=0.16.0"],
    },
    python_requires=">=3.8",
    include_package_data=True,
    license="MIT",
    url="https://github.com/ishans2404/axiom",
    project_urls={
        "Documentation": "https://github.com/ishans2404/axiom#readme",
        "Issues": "https://github.com/ishans2404/axiom/issues",
    },
)
