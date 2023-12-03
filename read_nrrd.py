import nrrd
import numpy as np
import skimage.measure  
import matplotlib.pyplot as plt


def extract_surface_mesh(id):
    filename_1 = f"data/corpus/example{id}_probCurve_1.nrrd"
    readdata_1, _ = nrrd.read(filename_1)
    corpus_callosum = readdata_1
    for i in range(2,5):
        filename_i = f"data/corpus/example{id}_probCurve_" + str(i) + ".nrrd"
        readdata_i, _ = nrrd.read(filename_i)
        corpus_callosum += readdata_i

    nrrd.write("data/corpus/corpus_callosum.nrrd", corpus_callosum)

    label_mask = corpus_callosum[97,:,:]
    label_mask = np.rot90(label_mask, k=-3)
    if id == 1:
        plt.imsave(f"data/corpus/corpus_callosum_{id}_slice_97.png", label_mask, cmap='gray')

    ls = skimage.measure.find_contours(label_mask, 900)
    vertices = np.vstack(ls)
    np.savetxt(f"data/corpus/corpus_callosum_{id}_vertices.txt", vertices)

extract_surface_mesh(id=1)
extract_surface_mesh(id=2)


