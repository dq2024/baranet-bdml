rm -rf CMakeCache.txt CMakeFiles/ Makefile cmake_install.cmake
cmake . -DCMAKE_BUILD_TYPE=Debug
make
python test_llama.py