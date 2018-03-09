
from distutils.core import setup

setup(
    name = 'sgdmf',
    version = '0.2.0',
    author = 'Tymoteusz Wolodzko',
    author_email = 'twolodzko+sgdmf@gmail.com',
    packages = ['sgdmf'],
    license = 'LICENSE.txt',
    description = 'Matrix factorization using stochastic gradient descent.',
    install_requires = [
        "numpy>=1.12.1",
        "scikit-learn>=0.19.1",
        "tqdm>=4.19.6"
    ],
)