import os,sys
import numpy as np

def merge_materials(material1, material2, alpha, gamma):
    result = material1*alpha + material2*(1-alpha)
    result = np.power(gamma)
    return result

