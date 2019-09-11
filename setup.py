import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="captcha-break",
    version="0.0.1",
    author="Author Name",
    author_email="someone@somewhere.com",
    description="Captcha Breaker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://example.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
