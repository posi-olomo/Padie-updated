from setuptools import setup, find_packages

setup(
    name="MyPadi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "django",
    ],
    include_package_data=True,
)
