sudo apt install -y build-essential swig
sudo apt install -y intel-mkl
sudo snap install cmake --classic

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> $HOME/.bashrc
source $HOME/.bashrc
conda update conda --yes
conda update --all --yes
conda install numpy --yes

git clone https://github.com/facebookresearch/faiss.git
cd faiss

cmake -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=ON \
    -DFAISS_OPT_LEVEL=avx2 \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=/home/ubuntu/miniconda/bin/python \
    -DCMAKE_BUILD_TYPE=Release .

make -C build -j faiss faiss_avx2

make -C build -j swigfaiss swigfaiss_avx2

(cd build/faiss/python && python setup.py install)

echo 'export PYTHONPATH=/home/ubuntu/faiss/build/faiss/python/build/lib:$PYTHONPATH' >> $HOME/.bashrc
source $HOME/.bashrc
