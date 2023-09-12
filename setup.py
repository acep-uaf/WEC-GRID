from setuptools import setup, find_packages

setup(
    name="WEC_GRID",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here, like:
        # 'numpy>=1.18.5',
        # 'pandas>=1.0.4',
    ],
    description="Description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your_email@example.com",
    url="URL to your project's repository",
    classifiers=[
        # Relevant classifiers: https://pypi.org/classifiers/
    ],
    include_package_data=True,
)
