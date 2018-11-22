@echo This file must be run at the same height as the pybind/opennn folder
@echo Running the installation of PyHamcrest.
start pip install pyhamcrest
PAUSE
cd %~dp0
cd opennn
@echo Compiling the OpenNN C ++ library, to link its code to Python
start python setup.py develop
PAUSE
@echo Running the installation of OpenNN.
cd ..
start pip install -e opennn
@echo Installation successful.