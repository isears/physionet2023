import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="physionet2023",
    version="0.0.1",
    author="Isaac Sears",
    author_email="isaac.j.sears@gmail.com",
    description="Physionet 2023 Challenge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["physionet2023"],
    url="https://github.com/isears/physionet2023",
    project_urls={
        "Bug Tracker": "https://github.com/isears/physionet2023/issues",
    },
)
