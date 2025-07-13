from setuptools import setup, find_packages

setup(
    name="ojkotka-agent-development-model",
    version="0.1.0",
    description="OJKotka Agent Development Model - Quantum Superagent Mood, Financial Optimization Framework, and Technical Specifications.",
    author="OJKotkaLKP",
    author_email="your.email@example.com",
    url="https://github.com/OJKotkaLKP/This-is-it",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "transformers>=4.12.0",
        "pandas>=1.3.0",
        "requests>=2.25.0",
        "pydantic>=1.8.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0"
    ],
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)