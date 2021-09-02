import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='neuron_dash',
    version='0.1',
    author="Abolfazl Ziaeemehr",
    author_email="a.ziaeemehr@gmail.com",
    description="Dashboard for neuron models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ziaeemehr/neuron_dash",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    # package_data={'sbi_nmms': ['DampOscillator.so']},
    install_requires=requirements,
    # include_package_data=True,
)
