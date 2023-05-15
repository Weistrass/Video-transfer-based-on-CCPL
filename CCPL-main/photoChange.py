import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import imageio
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import coral, calc_mean_std
from utils import makeVideo
from Loader import Dataset

import warnings
warnings.filterwarnings("ignore")


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def transform(vgg, decoder, SCT, content, style, alpha=1.0,
              interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    cF = vgg(content)
    sF = vgg(style)

    t = SCT(cF, sF)
    return t
def loadImg(imgPath, size):
    img = Image.open(imgPath).convert('RGB')
    transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()])
    return transform(img)

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str,
                    help='path to video frames')
parser.add_argument('--style_path', type=str,
                    help='the style source')
parser.add_argument('--testing_mode', default='art',
                    help='Artistic or Photo-realistic')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str,help='decoder to load')
parser.add_argument('--SCT', type=str,help='SCT module to load')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.mp4',
                    help='The extension name of the output video')
parser.add_argument('--output', type=str,
                    help='output dir')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
args = parser.parse_args()
def photochange(contentDirPath,stylepath,videoSavePath,mode,pictureDecoder,pictureSct):
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict.update({
        'content_dir':contentDirPath,
        'style_path':stylepath,
        'output':videoSavePath,
        'testing_mode':mode,
        'decoder':pictureDecoder,
        'SCT':pictureSct
    })
    #将字典转换回args对象
    print(args_dict)
    args = argparse.Namespace(**args_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # --content_video should be given.
    assert (args.content_dir)
    if args.content_dir:
        content_path = Path(args.content_dir)

    # --style_path should be given
    assert (args.style_path)
    if args.style_path:
        style_path = Path(args.style_path)


    styleV = loadImg(args.style_path, args.style_size).unsqueeze(0)

    content_dataset = Dataset(args.content_dir,
                              loadSize=args.content_size,
                              fineSize=args.content_size,
                              test=True,
                              video=True)
    content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
                                                 batch_size=1,
                                                 shuffle=False)
    # picture=[]
    # for i,(content,contentName) in enumerate(content_loader):
    #     contentName=contentName[0]
    #     picture.append(content.squeeze(0).float().numpy())
    # photo1=picture[2]
    # #photo1.transpose((1,2,0))
    # photo1.shape = (512,910,3)
    # cmax = np.max(photo1[:])
    # cmin = np.min(photo1[:])
    # pred_host = np.uint8((np.double(photo1)-cmin)/(cmax - cmin)*255)
    # img_file=Image.fromarray(pred_host,mode="RGB")
    #
    # img_file.save("output/picture/10.png",bitmap_format="png")
    decoder = net.decoder
    vgg = net.vgg
    network = net.Net(vgg, decoder, args.testing_mode)
    SCT = network.SCT

    SCT.eval()
    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    SCT.load_state_dict(torch.load(args.SCT))
    decoder = decoder if args.testing_mode == 'art' else nn.Sequential(*list(net.decoder.children())[10:])
    vgg = nn.Sequential(*list(vgg.children())[:31]) if args.testing_mode == 'art' else nn.Sequential(
        *list(vgg.children())[:18])

    vgg.to(device)
    decoder.to(device)
    SCT.to(device)

    contentV = torch.Tensor(1, 3, args.content_size, args.content_size)
    styleV = styleV.cuda()
    contentV = contentV.cuda()
    result_frames = []
    contents = []
    style = styleV.squeeze(0).cpu().numpy()
    sF = vgg(styleV)
    for i, (content, contentName) in enumerate(content_loader):
        print('Transfer frame %d...' % i)
        contentName = contentName[0]

        contentV.resize_(content.size()).copy_(content)
        contents.append(content.squeeze(0).float().numpy())
        # forward
        with torch.no_grad():
            gF = transform(vgg, decoder, SCT, contentV, styleV)
            transfer = decoder(gF)
        transfer = transfer.clamp(0, 1)
        result_frames.append(transfer.squeeze(0).cpu().numpy())

    makeVideo(contents, style, result_frames, args.output)
if __name__ == '__main__':
    contentframe = 'videoframe/frame2/'
    videostyle = 'input/style/in2.jpg'
    videosave = 'output'
    photoMode = 'photo'
    if photoMode == 'art':
        picturedecoder = 'artistic/decoder_iter_160000.pth.tar'
        picturesct = 'artistic/sct_iter_160000.pth.tar'
    else:
        picturedecoder = 'photo_realistic/decoder_iter_160000.pth.tar'
        picturesct = 'photo_realistic/sct_iter_160000.pth.tar'

    photochange(contentframe, videostyle, videosave, photoMode, picturedecoder, picturesct)
