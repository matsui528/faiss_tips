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
Note that `index.add` function copies the data. If you would like to reduce the overhead of the copy, see this [wiki](https://github.com/facebookresearch/faiss/wiki/Brute-force-search-without-an-index) and [gists](https://gist.github.com/mdouze/8e47d8a5f28280df7de7841f8d77048d).

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
Or, you can serialize the index into binary array (np.array). You can save/load it via numpy IO functions.
```python
chunk = faiss.serialize_index(index)
np.save("index.npy", chunk)
index3 = faiss.deserialize_index(np.load("index.npy"))   # identical to index
```
You can even use pickle:
```python
import pickle
with open("index.pkl", "wb") as f:
    pickle.dump(chunk, f)
with open("index.pkl", "rb") as f:
    index4 = faiss.deserialize_index(pickle.load(f))   # identical to index
```

## k-means (CPU/GPU)
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
# For GPU(s), run the following line. This will use all GPUs
# kmeans = faiss.Kmeans(d=D, k=K, niter=20, verbose=True, gpu=True)

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

code_sz = index.invlists.code_size
# Code size per vector in bytes. This equals to
# - 4 * D  (if IndexIVFFlat)
# - M      (if IndexIVFPQ with nbits=8)
print("code_sz:", code_sz)

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

## Hamming distance table
Given two sets of binary vectors, pairwise hamming distances can be computed as follows.

The code is from [https://github.com/facebookresearch/faiss/issues/740#issuecomment-470959391](https://github.com/facebookresearch/faiss/issues/740#issuecomment-470959391)

```python
def pairwise_hamming_dis(a, b):
    """ compute the pairwise Hamming distances between two matrices """
    na, d = a.shape
    nb, d2 = b.shape
    assert d == d2

    dis = np.empty((na, nb), dtype='int32')

    faiss.hammings(
        faiss.swig_ptr(a), faiss.swig_ptr(b),
        na, nb, d,
        faiss.swig_ptr(dis)
    )
    return dis

# Each vector must be the form of "uint8 * ncodes",
# where ncodes % 8 == 0
xq = np.array([[0, 0, 0, 0, 0, 0, 0, 2],     # [0, 0, ..., 1, 0] <- 64 bits (ncodes=8)
               [0, 0, 0, 0, 0, 0, 0, 3]],    # [0, 0, ..., 1, 1]
               dtype=np.uint8)
xb = np.array([[0, 0, 0, 0, 0, 0, 0, 2],     # [0, 0, ..., 1, 0]
               [0, 0, 0, 0, 0, 0, 0, 0],     # [0, 0, ..., 0, 0]
               [0, 0, 0, 0, 0, 0, 0, 1]],    # [0, 0, ..., 0, 1]
               dtype=np.uint8)

dis = pairwise_hamming_dis(xq, xb)
print(dis)
# [[0 1 2]
#  [1 2 1]]
```


## Merge results
You can merge search results from several indices by [ResultHeap](https://github.com/facebookresearch/faiss/blob/master/faiss/python/__init__.py#L966)

```python
import faiss
import numpy as np

D = 128
N = 10000
Nq = 3
X = np.random.random((N, D)).astype(np.float32)
Xq = np.random.random((Nq, D)).astype(np.float32)

# Setup
index = faiss.IndexFlatL2(D)
index.add(X)

# Search
topk = 10
dists, ids = index.search(x=Xq, k=topk)
print("dists:", dists)
print("ids:", ids)


# Setup with two indices
index1 = faiss.IndexFlatL2(D)
index1.add(X[:2000])   # Store the first 2000 vectors
index2 = faiss.IndexFlatL2(D)
index2.add(X[2000:])   # Store the remaining

# Search for both indices
dists1, ids1 = index1.search(x=Xq, k=topk)
dists2, ids2 = index2.search(x=Xq, k=topk)

# Merge results
result_heap = faiss.ResultHeap(nq=Nq, k=topk)
result_heap.add_result(D=dists1, I=ids1)
result_heap.add_result(D=dists2, I=ids2 + 2000)  # 2000 is an offset
result_heap.finalize()
print("dists:", result_heap.D)
print("ids:", result_heap.I)

assert np.array_equal(dists, result_heap.D)
assert np.array_equal(ids, result_heap.I)
```

## `std::vector` to/from `np.array`
When you would like to directly handle std::vector in the c++ class, you can convert std::vector to np.array by `faiss.vector_to_array`. Similarily, you can call `faiss.copy_array_to_vector` to convert np.array to std::vector. See [python/faiss.py](https://github.com/facebookresearch/faiss/blob/master/python/faiss.py) for more details.

```python
import faiss
import numpy as np

D = 2
N = 3
X = np.random.random((N, D)).astype(np.float32)
print(X)
# [[0.8482132  0.17902061]
#  [0.07226888 0.15747449]
#  [0.41783017 0.9381101 ]]

# Setup
index = faiss.IndexFlatL2(D)
index.add(X)

# Let's see std::vector inside the c++ class (IndexFlatL2)
# We cannot directly read/update them by nparray.
print(type(index.xb))  # <class 'faiss.swigfaiss.FloatVector'>
print(index.xb)  # <faiss.swigfaiss.FloatVector; proxy of <Swig Object of type 'std::vector< float > *' at 0x7fd62822a690> >

# Convert std::vector to np.array
xb_np = faiss.vector_to_array(index.xb)  # std::vector -> np.array
print(type(xb_np))  # <class 'numpy.ndarray'>
print(xb_np)  # [0.8482132  0.17902061 0.07226888 0.15747449 0.41783017 0.9381101 ]

# We can also convert np.array to std::vector
X2 = np.random.random((N, D)).astype(np.float32)  # new np.array
X2_std_vector = faiss.FloatVector()  # Buffer.
print(type(X2_std_vector))  # <class 'faiss.swigfaiss.FloatVector'>
faiss.copy_array_to_vector(X2.reshape(-1), X2_std_vector)  # np.array -> std::vector. Don't forget to flatten the np.array

# The obtained std:vector can be used inside the c++ class
index.xb = X2_std_vector  # You can set this in IndexFlatL2 (Be careful. This is dangerous)

print(index.search(x=X2, k=3))  # Works correctly
# (array([[0.        , 0.09678471, 0.5692644 ],
#        [0.        , 0.09678471, 0.23682791],
#        [0.        , 0.23682791, 0.5692644 ]], dtype=float32), array([[0, 1, 2],
#        [1, 0, 2],
#        [2, 1, 0]]))
```
