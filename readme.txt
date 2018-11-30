INSTALL:

Install Visual Studio 2015 or greater in Windows or gcc in Linux. 
Install Python (Anaconda recommended)
Install Cmake

pip install PyHamcrest

pip install twine

Open Anaconda cmd, 

python setup.py develop

pip install -e opennn


DEPLOY:

Cambiar en Setup.py la versión a una superior de la librería a subir.

Ejecutar en la consola: python setup.py sdist bdist_wheel

Ejecutar en la consola: twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
Pedirá usuario y contraseña, está en keepass.