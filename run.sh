export PYBIND11_INC="$(python3 -c 'import pybind11; print(pybind11.get_include())')"
export PYBIND11_CMAKE_DIR="$(python3 -m pybind11 --cmakedir)"
export CMAKE_PREFIX_PATH="$PYBIND11_CMAKE_DIR:${CMAKE_PREFIX_PATH}"
export CPLUS_INCLUDE_PATH="$PYBIND11_INC:${CPLUS_INCLUDE_PATH}"
export CPATH="$PYBIND11_INC:${CPATH}"

# Install requirements if not already present (optional but good for safety)
# pip install -r requirements.txt

rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Debug
make

cd ..
python3 test_llama.py