
from photoChange import *
contentframe='videoframe/frame2/'
videostyle='input/style/in2.jpg'
videosave='output'
photoMode='photo'
if photoMode=='art':
    picturedecoder='artistic/decoder_iter_160000.pth.tar'
    picturesct='artistic/sct_iter_160000.pth.tar'
else:
    picturedecoder='photo_realistic/decoder_iter_160000.pth.tar'
    picturesct='photo_realistic/sct_iter_160000.pth.tar'

photochange(contentframe,videostyle,videosave,photoMode,picturedecoder,picturesct)

