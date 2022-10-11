#! /usr/bin/env bash

# from https://gist.github.com/mdouze/2b96e804ee2825e72323c0a296061c7e

# get script directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  # if $SOURCE was a relative symlink, we need to resolve it
  # relative to the path where the symlink file was located
  [[ $SOURCE != /* ]] && SOURCE="$SCRIPT_DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

CURRENT=$(pwd)

cd "${SCRIPT_DIR}/.."

conda install -y cython numpy swig cmake

if [ ! -d "faiss" ]; then
    git clone https://github.com/facebookresearch/faiss.git
fi

cd faiss
git pull # update to latest version
cmake -B build -DBUILD_TESTING=ON -DFAISS_ENABLE_GPU=OFF \
     -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_C_API=ON \
     -DPython_EXECUTABLE=$(which python)
make -k -C build -j faiss
# C++ tests do not work
# make -C build test
make -C build -j swigfaiss
cd build/faiss/python
python3 setup.py build
cd -
cd tests
PYTHONPATH=../build/faiss/python/build/lib/ python3 -m unittest discover

cd "${CURRENT}"
