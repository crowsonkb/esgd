import setuptools

setuptools.setup(
    name='esgd',
    version='0.1',
    description='ESGD-M is a stochastic non-convex second order optimizer, suitable for training deep learning models, for PyTorch.',
    # TODO: add long description
    long_description='ESGD-M is a stochastic non-convex second order optimizer, suitable for training deep learning models, for PyTorch.',
    url='https://github.com/crowsonkb/esgd',
    author='Katherine Crowson',
    author_email='crowsonkb@gmail.com',
    license='MIT',
    packages=['esgd'],
    install_requires=['torch'],
    python_requires=">=3.6",
    # TODO: Add classifiers
    classifiers=[],
)
