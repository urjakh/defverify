from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines()]


setup(
    name="hs_generalization",
    version="1.0",
    author="Urja Khurana",
    author_email="u.khurana@vu.nl",
    url="",
    description="Code for investigating the influence of hate speech definitions on models' capability to generalize.",
    packages=find_packages(),
    install_requires=requirements,
)
