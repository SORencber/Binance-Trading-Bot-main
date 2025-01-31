# setup.py

from setuptools import setup, find_packages

setup(
    name="advanced_bot2",              # Proje adınız
    version="0.1.0",
    packages=find_packages(),          # Otomatik olarak __init__.py içeren klasörleri bulur
    include_package_data=True,         # data dosyaları vs. için
    install_requires=[
        # Burada proje bağımlılıklarını yazabilirsiniz, örn. "pandas", "numpy", vb.
    ],
    description="Binance Trading Bot - advanced_bot2",
    author="YourName",
    author_email="youremail@example.com",
    url="https://github.com/... or your URL",
    python_requires=">=3.8",           # Python sürüm kısıtı
)
