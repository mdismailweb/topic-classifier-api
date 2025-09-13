from setuptools import setup, find_packages

setup(
    name="topic-classifier-api",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'Flask==2.0.1',
        'gunicorn==20.1.0',
        'scikit-learn==0.24.2',
        'numpy==1.19.5',
        'pandas==1.3.0',
        'joblib==1.0.1',
        'Werkzeug==2.0.3'
    ],
    python_requires='>=3.7,<3.10',
)