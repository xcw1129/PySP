from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pysp-xcw",  # 包名
    version="7.2.0",      # 版本号
    author="Xiong Chengwen",   # 作者
    author_email="xcw1824@outlook.com",  # 作者邮箱
    description="Various classical signal analysis and processing algorithms, in the field of mechanical fault diagnosis, are implemented.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xcw1129/PySP",  # 项目主页
    packages=find_packages(),  # 自动发现包
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Intended Audience :: Science/Research",
        "Development Status :: 5 - Production/Stable",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
    ],
    include_package_data=True,
    keywords=["signal processing", "fault diagnosis", "time-frequency analysis", "spectral analysis"],
    project_urls={
        "Bug Tracker": "https://github.com/xcw1129/PySP/issues",
        "Source Code": "https://github.com/xcw1129/PySP",
    },
)