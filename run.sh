export PYBIND11_INC="$(python -c 'import pybind11; print(pybind11.get_include())')"
export PYBIND11_CMAKE_DIR="$(python -m pybind11 --cmakedir)"
export CMAKE_PREFIX_PATH="$PYBIND11_CMAKE_DIR:${CMAKE_PREFIX_PATH}"
export CPLUS_INCLUDE_PATH="$PYBIND11_INC:${CPLUS_INCLUDE_PATH}"
export CPATH="$PYBIND11_INC:${CPATH}"

rm -rf CMakeCache.txt CMakeFiles/ Makefile cmake_install.cmake
cmake . -DCMAKE_BUILD_TYPE=Debug
make
python test_llama.py