from setuptools import find_packages, setup


with open("README.md") as f:
    long_description = f.read()

INSTALL_REQUIRES = [
    "loguru==0.7.2",
    "tqdm==4.66.1",
    "matplotlib==3.8.0",
    "numpy==1.24.1",
    "optuna==3.4.0",
    "joblib==1.3.2",
    "pandas==2.1.1",
    "scikit-learn==1.3.1",
    "xgboost==2.0.1",
    "lightgbm==4.1.0",
    "catboost==1.2.2",
]

if __name__ == "__main__":
    setup(
        name="all you need is ml",
        description="all you need is ml",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Minseong Kim",
        author_email="qtddpms@gmail.com",
        url="https://github.com/mskimS2/tabular-data",
        license="Apache 2.0",
        package_dir={"": "src"},
        packages=find_packages("src"),
        entry_points={},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        platforms=["linux", "unix"],
        python_requires=">=3.8",
    )