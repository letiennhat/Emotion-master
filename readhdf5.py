import h5py
filename = "models/emotion_model.hdf5"
with h5py.File(filename,"r") as f:
    print("Keys : %s" %f.keys())
    a_group_key = list(f.keys())[0]
    data = list(f[a_group_key])
    print(data)
