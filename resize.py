from glob import glob
from PIL import Image
import os

# This code is for resizing image.
# For example, changing the 2000*200 to be 1000*100. By Tung-Te Lin, 18th October 2019.

test_figure_origin = glob('test_figure_2000_200/*.png')
training_figure_origin = glob('training_figure_2000_200/*.png')
validation_figure_origin = glob('validation_figure_2000_200/*.png')

test_image_origin = glob('test_image_origin/*.png')
training_image_origin = glob('training_image_origin/*.png')
validation_image_origin = glob('validation_image_origin/*.png')


##for path in test_image_origin:
##    im = Image.open(path)
##    im = im.convert('L')
##    filename = os.path.basename(path)
##    im.save('test_image/'+filename)


##for path in training_image_origin:
##    im = Image.open(path)
##    im = im.convert('L')
##    filename = 'training_image/' + os.path.basename(path)
##    im.save(filename)
##
##    #os.remove(filename)
##
##for path in validation_image_origin:
##    im = Image.open(path)
##    im = im.convert('L')
##    filename = 'validation_image/' + os.path.basename(path)
##    im.save(filename)



##for path in test_figure_origin:
##    im = Image.open(path)
##    im = im.convert('L')
##    im = im.resize((1000,100))
##    filename = os.path.basename(path)
##    im.save('test_figure/'+filename)


for path in training_figure_origin:
    im = Image.open(path)
    im = im.convert('L')        # This line is setting gray scale mode.
    im = im.resize((1000,100))
    filename = 'training_figure/' + os.path.basename(path)
    im.save(filename)

    #os.remove(filename)

for path in validation_figure_origin:
    im = Image.open(path)
    im = im.convert('L')
    im = im.resize((1000,100))
    filename = 'validation_figure/' + os.path.basename(path)
    im.save(filename)

    #os.remove(filename)

