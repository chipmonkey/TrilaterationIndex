import h5py

f = h5py.File('fashion-mnist-784-euclidean.hdf5', 'r')
print(list(f.keys()))
