from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pysp-xcw",  # 包名
    version="7.5.1",  # 版本号
    author="Xiong Chengwen",  # 作者
    author_email="xcw1824@outlook.com",  # 作者邮箱
    description="Various classic and modern signal analysis and processing algorithms are implemented.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xcw1129/PySP",  # 项目主页
    packages=find_packages(),  # 自动发现包
    package_data={
        "PySP._Assist_Module": ["times+simsun.ttf"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    keywords=[
        "signal processing",
        "spectral analysis",
        "time-frequency analysis",
        "feature extraction",
        "signal decomposition",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/xcw1129/PySP/issues",
        "Source Code": "https://github.com/xcw1129/PySP",
    },
)
