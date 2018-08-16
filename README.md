# Faiss tips
- Some useful tips for [faiss](https://github.com/facebookresearch/faiss)
- We suppose faiss is installed via conda:
    ```
    conda install faiss-cpu -c pytorch
    conda install faiss-gpu -c pytorch
    ```

## Nearest neighbor search (CPU)
The most basic nearest neighbor search by L2 distance. This is much faster than scipy. You should first try this, especially when the database is relatively small (N<10^6). The search is automatically paralellized.
```python
import faiss
import numpy as np

D = 128
N = 10000
X = np.random.random((N, D)).astype(np.float32)  # inputs of faiss must be float32

# Setup
index = faiss.IndexFlatL2(D)
index.add(X)

# Search
topk = 4
dists, ids = index.search(x=X[:3], k=topk)  # Use the top three vectors for queyring
print(type(dists), dists.dtype, dists.shape)  # <class 'numpy.ndarray'> float32 (3, 4)
print(type(ids), ids.dtype, ids.shape)  # <class 'numpy.ndarray'> int64 (3, 4)

# Show params
print("N:", index.ntotal)
print("D:", index.d)
```

## Nearest neighbor search (GPU)
The nearest neighbor search on GPU(s). This returns the (almost) same result as that of CPU. This is extremely fast although there aren't any approximation steps. You should try this if the data fit into your GPU memory.
```python
import faiss
import numpy as np
import os

D = 128
N = 10000
X = np.random.random((N, D)).astype(np.float32)  # inputs of faiss must be float32

# GPU config
gpu_ids = "0"  # can be e.g. "3,4" for multiple GPUs 
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

# Setup
cpu_index = faiss.IndexFlatL2(D)
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
gpu_index.add(X)

# Search
topk = 4
dists, ids = gpu_index.search(x=X[:3], k=topk)
```


## Approximate nearest neighbor search (CPU)
There are several methods for ANN search. Currently, HNSW + IVFPQ achieves the best performance for billion-scale data in terms of the balance among memory, accuracy, and runtime.
```python
import faiss
import numpy as np

D = 128
N = 1000000
Xt = np.random.random((10000, D)).astype(np.float32)  # 10000 vectors for training
X = np.random.random((N, D)).astype(np.float32)

# Param of PQ
M = 16  # The number of sub-vector. Typically this is 8, 16, 32, etc.
nbits = 8 # bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte
# Param of IVF
nlist = 1000  # The number of cells (space partition). Typical value is sqrt(N)
# Param of HNSW
hnsw_m = 32  # The number of neighbors for HNSW. This is typically 32

# Setup
quantizer = faiss.IndexHNSWFlat(D, hnsw_m)
index = faiss.IndexIVFPQ(quantizer, D, nlist, M, nbits)

# Train
index.train(Xt)

# Add
index.add(X)

# Search
index.nprobe = 8  # Runtime param. The number of cells that are visited for search.
topk = 4
dists, ids = index.search(x=X[:3], k=topk)

# Show params
print("D:", index.d)
print("N:", index.ntotal) 
print("M:", index.pq.M)
print("nbits:", index.pq.nbits)
print("nlist:", index.nlist)
print("nprobe:", index.nprobe)
```

Note that you might need to set `quantizer` and `index` as member variables if you'd like to use them inside your class, e.g.,
```
self.quantizer = faiss.IndexHNSWFlat(D, hnsw_m)
self.index = faiss.IndexIVFPQ(self.quantizer, D, nlist, M, nbits)
```
See https://github.com/facebookresearch/faiss/issues/540

## I/O
Faiss index can be read/write via util functions:
```python
faiss.write_index(index, "index.bin")
index2 = faiss.read_index("index.bin")  # index2 is identical to index
```


## k-means (CPU)
k-means in faiss is much faster than that in sklearn. See the [benchmark](https://github.com/DwangoMediaVillage/pqkmeans/blob/master/tutorial/4_comparison_to_faiss.ipynb).
```python
import faiss
import numpy as np

D = 128
N = 10000
K = 10  # The number of clusters
X = np.random.random((N, D)).astype(np.float32)

# Setup
kmeans = faiss.Kmeans(d=D, k=K, niter=20, verbose=True)

# Run clustering
kmeans.train(X)

# Error for each iteration
print(kmeans.obj)  # array with 20 elements

# Centroids after clustering
print(kmeans.centroids.shape)  # (10, 128)

# The assignment for each vector.
dists, ids = kmeans.index.search(X, 1)  # Need to run NN search again
print(ids.shape)  # (10000, 1)

# Params
print("D:", kmeans.d)
print("K:", kmeans.k)
print("niter:", kmeans.cp.niter)
```

## Get posting lists
Given an IVF type index and queries, you can check the nearest posting lists by calling the search function in a quantizer:
```python
# Suppose index = faiss.indexIVF...
dists, ids = index.quantizer.search(x=X[:3], k=1)
print(dists) 
print(ids)  # e.g., ids[2] is the ID of the nearest posting list for the third query (X[2])
```
Note that you can enumerate all posting lists (both indices and codes) if you want:
```python
id_poslists = []  
code_poslists = []  

code_sz = index.invlists.code_size  # Code size per vector in bytes. This equals to M if nbits=8

for list_no in range(index.nlist):
    list_sz = index.invlists.list_size(list_no)  # The length of list_no-th posting list
    
    # Fetch
    id_poslist = np.array(faiss.rev_swig_ptr(index.invlists.get_ids(list_no), list_sz))
    code_poslist = np.array(faiss.rev_swig_ptr(index.invlists.get_codes(list_no), list_sz * code_sz))
    
    id_poslists.append(id_poslist)
    code_poslists.append(code_poslist)
    
print(len(id_poslists)) # nlist
print(len(id_poslists[3]))  # e.g., 463
print(id_poslists[3].dtype)  # int64
print(id_poslists[3][:4])  # e.g., [ 309 1237 6101]

print(len(code_poslists))  # nlist
print(len(code_poslists[3]))  # e.g., 7408  (i.e., code_sz=16)
print(code_poslists[3].dtype)  # uint8
print(code_poslists[3][:4 * code_sz])  # e.g., [ 36  66 242  ... 126]
```


## Multiple/Single threads
Faiss usually uses multiple threads automatically via openmp for parallel computing. If you would like to use a single thread only, run this line in the beginning.
```python
faiss.omp_set_num_threads(1)
```
Or, you can explicitly set an environment variable in your terminal.
```
export OMP_NUM_THREADS=1
```

