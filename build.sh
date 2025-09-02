sudo apt install -y build-essential swig
sudo apt install -y intel-mkl
sudo snap install cmake --classic

cd $HOME
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
bash $HOME/miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> $HOME/.bashrc
conda update conda --yes
conda update --all --yes
conda install numpy --yes

cd $HOME
git clone https://github.com/facebookresearch/faiss.git
cd faiss

cmake -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=ON \
    -DFAISS_OPT_LEVEL=avx2 \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DPython_EXECUTABLE=$HOME/miniconda/bin/python \
    -DCMAKE_BUILD_TYPE=Release .

if [ $? -ne 0 ]; then
    echo "Error: CMake build failed"
    exit 1
fi

make -C build -j faiss faiss_avx2

if [ $? -ne 0 ]; then
    echo "Error: Building C++ library failed"
    exit 1
fi

make -C build -j swigfaiss swigfaiss_avx2

if [ $? -ne 0 ]; then
    echo "Error: Building Python bindings failed"
    exit 1
fi

cd build/faiss/python
python setup.py install

export PYTHONPATH=$HOME/faiss/build/faiss/python/build/lib:$PYTHONPATH
echo 'export PYTHONPATH=$HOME/faiss/build/faiss/python/build/lib:$PYTHONPATH' >> $HOME/.bashrc
