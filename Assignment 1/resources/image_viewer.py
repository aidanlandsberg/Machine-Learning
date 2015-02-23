"""
A useful image viewer. Using the widgets of ipython notebook.
Assume that the images to be displayed are in an array ic

Input
-----
ic : (n,i,j) ndarray
   n number of images
   Each image is (i,j).
   
"""
from IPython.html.widgets import interact

def view_image(n=0):
    plt.imshow(ic[n], cmap='gray', interpolation='nearest')
    plt.show()

w = interact(view_image, n=(0, len(ic)-1))

