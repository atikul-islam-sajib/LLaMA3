from setuptools import setup, find_packages

setup(
    name="LLaMA3",
    version="0.0.1",
    description="""
    About
    -----
    LLaMA3: A minimal and educational PyTorch implementation of Meta's LLaMA 3 architecture.
    This project includes Grouped Query Attention (GQA), RMSNorm, Rotary Positional Embeddings (RoPE),
    SwiGLU feedforward, and full decoder-only Transformer design based on LLaMA 3 specifications.
    """,
    author="Atikul Islam Sajib",
    author_email="atikulislamsajib137@gmail.com",
    url="https://github.com/atikul-islam-sajib/LLaMA3",
    packages=find_packages(),
    install_requires=[
        "dvc",
        "dagshub",
        "graphviz",
        "matplotlib",
        "numpy",
        "mlflow",
        "PyYAML",
        "torch",
        "torchview",
        "torchaudio",
        "torchvision",
        "torchsummary",
        "torchmetrics",
        "scikit-image",
        "opencv-python",
        "nltk",
        "python-dotenv",
        "ipykernel",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    keywords="LLaMA3, GQA, transformer, deep-learning, language-model, PyTorch",
    project_urls={
        "Bug Tracker": "https://github.com/atikul-islam-sajib/LLaMA3/issues",
        "Documentation": "https://github.com/atikul-islam-sajib/LLaMA3",
        "Source Code": "https://github.com/atikul-islam-sajib/LLaMA3",
    },
)
