"""speechvis.py is just a helper module to import some classes and methods. Invoke it using `ipython -i speechvis.py`
or an equivalent command."""

from ivfcrvis.segment import *
from ivfcrvis.recording import *
from ivfcrvis.mfcc_pca import plot_mfcc_pca
from matplotlib.pyplot import *

root = 'C:/Users/Public/Documents/ivfcr'
ids = ['e20160102_095210_010584',
       'e20160413_154936_010580',
       'e20160711_124830_010580']

if __name__ == "__main__":
    ion()
    print("IVFCR Root: {0} (Use `root=<value>` to change)".format(root))
    print("Ids: {0} (Use `ids.append(<value>)` to add)".format(ids))
