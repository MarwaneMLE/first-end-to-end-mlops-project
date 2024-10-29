from setuptools import find_packages, setup
from typing import List


setup(
    name="DimondPricePrediction",
    version="0.0.1",
    author="MarwawneMLE",
    author_email = "khmarwane10@gmail.com",
    install_requires=["scikit-learn", "pandas","numpy"],
    pachages = find_packages()
)