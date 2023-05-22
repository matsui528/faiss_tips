# Build
This document is a step-by-step instruction for building faiss from the source. We assume:
- x86 architecture
- CPU
- Ubuntu 22.04 
- miniconda for python environment
- Intel MKL (we can install it simply by `apt` for Ubuntu 20.04 or higher)
- AVX2

We will install faiss and conda on `$HOME`, i.e., 
```console
/home/ubuntu
├── faiss
└── miniconda
```
You can always change the structure.

We tested the build process on an AWS EC2 c5.12xlarge instance.

Official documents:
- [Official installation guide](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md)
- [Official wiki](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss)
- [Official conda config](https://github.com/facebookresearch/faiss/tree/master/conda)

## tl;dr
- [build script](build.sh)
- github actions
    - [code](.github/workflows/build_from_source.yml)
    - [result](https://github.com/matsui528/faiss_tips/actions/workflows/build_from_source.yml)


## Prerequisite

### g++ and swig
```bash
sudo apt install -y build-essential swig
```

### BLAS: Intel MKL
Installing Intel MKL has been extremely hard. Fortunately, for Ubuntu 20.04 or higher, we can install it simply by `apt install`.
```bash
sudo apt install -y intel-mkl
```
You may be asked about the license. Please carefully understand the terms of the license and choose yes to the question of "Use libmkl_rt.so as the default alternative to BLAS/LAPACK? ".

Note that the official wiki introduces [the way to use MKL inside the anaconda](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss). I've tried it dozens of times, and it doesn't work... If anyone can make it work, please send me an issue/PR.

If you cannot install intel-mkl, you can use open-blas by `sudo apt install -y libopenblas-dev`


### cmake
Currently, cmake from apt is old (3.16 for Ubuntu 20.04, and 3.22 for Ubuntu 22.04). For faiss 1.7+, we need cmake 3.23+. There are three options to install new cmake.
- Build from source
- Install by snap. This is the easiest.
    ```bash
    sudo snap install cmake --classic
    ```
    Note that WSL recently supported snap. See [this](https://devblogs.microsoft.com/commandline/systemd-support-is-now-available-in-wsl/#set-the-systemd-flag-set-in-your-wsl-distro-settings).
- If you've installed conda, you can install cmake by conda.
    ```bash
    conda install -c anaconda cmake 
    ```
- [Use APT repository](https://askubuntu.com/a/1157132). Seems easy. Not tested by myself though.


### miniconda
We will use miniconda for python. See [this](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html#install-macos-silent) for the instruction of the silent installation.
```bash
cd $HOME
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
bash $HOME/miniconda.sh -b -p $HOME/miniconda
```
Then activate the miniconda
```bash
export PATH="$HOME/miniconda/bin:$PATH"
```
Write the above line at bashrc so we don't have to run it every time.
```bash
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> $HOME/.bashrc
```
Install required packages
```bash
conda update conda --yes
conda update --all --yes
conda install numpy --yes
```
Make sure your python path works.
```bash
which python    # e.g., /home/ubuntu/miniconda/bin/python
```


## Build
Clone the repo.
```bash
cd $HOME
git clone https://github.com/facebookresearch/faiss.git
cd faiss
```
Run cmake. See the [official instruction](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md#step-1-invoking-cmake) for the explanation of each option

```bash
cmake -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=ON \
    -DFAISS_OPT_LEVEL=avx2 \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DPython_EXECUTABLE=$HOME/miniconda/bin/python \
    -DCMAKE_BUILD_TYPE=Release .
```
For `-DPython_EXECUTABLE`, write the output of `which python`.
This `cmake` creates a `build` directory.
Note that you don't need to specify `-DBLA_VENDOR` and `-DMKL_LIBRARIES`.

In the log message, you will find that the cmake correctly located the MKL: `-- Found MKL: /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so;/usr/lib/x86_64-linux-gnu/libmkl_sequential.so;/usr/lib/x86_64-linux-gnu/libmkl_core.so;-lpthread;-lm;-ldl`


Then, run make to build the library.
```bash
make -C build -j faiss faiss_avx2
```
This will create `build/faiss/libfaiss.so` and `build/faiss/libfaiss_avx2.so`. I'm not sure about this part, but we need to specify `faiss_avx2` as well manually.

Let's check the link information by:
```bash
ldd build/faiss/libfaiss_avx2.so
```
This will show something like:
```bash
        linux-vdso.so.1 (0x00007ffc6dcc7000)
        libmkl_intel_lp64.so => /lib/x86_64-linux-gnu/libmkl_intel_lp64.so (0x00007f4e3cfd1000)
        libmkl_sequential.so => /lib/x86_64-linux-gnu/libmkl_sequential.so (0x00007f4e3b9b9000)
        libmkl_core.so => /lib/x86_64-linux-gnu/libmkl_core.so (0x00007f4e37699000)
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f4e37676000)
        libgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007f4e37634000)
        libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f4e37450000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f4e37301000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f4e372e6000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f4e370f4000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f4e3de2a000)
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f4e370ee000)
```
Here, you can see `/lib/x86_64-linux-gnu/libmkl_intel_lp64.so`, etc. This message means that faiss links the system-installed Intel MKL correctly.


Then let's test c++. It seems `make -C build test` doesn't work. So let's try `demo_ivfpq_indexing`

```bash
make -C build -j demo_ivfpq_indexing
./build/demos/demo_ivfpq_indexing
```
It takes 7 sec for AWS EC2 c5.12xlarge: `[7.298 s] Query results (vector ids, then distances):`.

Note that `demo_ivfpq_indexing` uses `libfaiss.so`. If you want to use `libfaiss_avx2.so`, please rewrite `target_link_libraries(demo_ivfpq_indexing PRIVATE faiss)` to `target_link_libraries(demo_ivfpq_indexing PRIVATE faiss_avx2)` in `$HOME/faiss/demos/CMakeLists.txt`.


Then let's build the python module. Run the following.
```bash
make -C build -j swigfaiss swigfaiss_avx2
```
This will create files on `build/faiss/python`.

Then let's install the module on your python.
```bash
cd build/faiss/python
python setup.py install
```
This will update your python environment (You can uninstall it by `pip uninstall faiss`).

Finally, you need to specify the PYTHONPATH. Activate it, and write it on `~/.bashrc`.
```bash
export PYTHONPATH=$HOME/faiss/build/faiss/python/build/lib:$PYTHONPATH
echo 'export PYTHONPATH=$HOME/faiss/build/faiss/python/build/lib:$PYTHONPATH' >> $HOME/.bashrc
```

Now you can use faiss from python.
Let's check it.
```bash
cd    # Recommend changing the directory. We need to make sure that we can use python-faiss from any place
python -c "import faiss, numpy; err = faiss.Kmeans(10, 20).train(numpy.random.rand(1000, 10).astype('float32')); print(err)"
```
You will see something like `483.5049743652344`.


## Check AVX2 is working or not
Let's check AVX2 is activated or not.

```bash
cd
LD_DEBUG=libs python -c "import faiss" 2>&1 | grep libfaiss.so
```
If you see something, then your AVX2 **is not** activated.

Run the following as well
```bash
cd
LD_DEBUG=libs python -c "import faiss" 2>&1 | grep libfaiss_avx2.so
```
If you see something, then your AVX2 **is** activated.

To actually evaluate the runtime, please save the following as `check.py`.
This code compares `IndexPQ` and `IndexPQFastScan`. Here, `IndexPQFastScan` is a faster (approximated) version of `IndexPQ` with SIMD instructions (AVX2 for usual x86 computers).
```python
import faiss
import numpy as np
import time

np.random.seed(234)
D = 128
N = 10000
X = np.random.random((N, D)).astype(np.float32)
M = 64
nbits = 4

pq = faiss.IndexPQ(D, M, nbits)
pq.train(X)
pq.add(X)

pq_fast = faiss.IndexPQFastScan(D, M, nbits)
pq_fast.train(X)
pq_fast.add(X)

t0 = time.time()
d1, ids1 = pq.search(x=X[:3], k=5)
t1 = time.time()
print(f"pq: {(t1 - t0) * 1000} msec")

t0 = time.time()
d2, ids2 = pq_fast.search(x=X[:3], k=5)
t1 = time.time()
print(f"pq_fast: {(t1 - t0) * 1000} msec")

assert np.allclose(ids1, ids2)
```

Then run `python check.py`.
If AVX2 is properly activated, pq_fast should be roughly 10x faster:
```bash
pq: 1.8916130065917969 msec
pq_fast: 0.1723766326904297 msec
```


## (Advanced) ARM
- For ARM architecture such as AWS Graviton2, you can build faiss by rewriting some of the above instructions as follows.
- For SIMD, we'll use NEON instead of AVX2.

### BLAS: BLAS-openmp
We cannot install Intel MKL for ARM by apt. So an easy way is to use openblas-openmp.
```bash
sudo apt install -y libopenblas-openmp-dev
```

### miniconda
Replace the path to the bash file with the one for arm:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O $HOME/miniconda.sh
bash $HOME/miniconda.sh -b -p $HOME/miniconda
```

### Build 
To make the library, you don't need to specify avx2.
```bash
cmake -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=ON \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DPython_EXECUTABLE=$HOME/miniconda/bin/python \
    -DCMAKE_BUILD_TYPE=Release .
```

You don't need {faiss, swigfaiss}_avx2
```bash
make -C build -j faiss
make -C build -j swigfaiss
```
