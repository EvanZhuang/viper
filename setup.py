from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires= [
    "evaluate",
    "gradio",
    "huggingface_hub",
    "nltk",
    "numpy",
    "packaging",
    "pandas",
    "Pillow",
    "requests",  # Note: use lowercase for 'requests'
    "scipy",
    "timm",
    "tokenizers",
    "torch",
    "tqdm",
    "transformers",
],

setup(  
    name="vipervlm",
    version="0.1.0",
    author="Yufan Zhuang, Pierce Chuang, Yichao Lu, Abhay Harpale, Vikas Bhardwaj, Jingbo Shang",
    author_email="y5zhuang@ucsd.edu",
    description="PyTorch Implementation for Viper: Open Mamba-based Vision-Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EvanZhuang/viper",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.7.0',
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)