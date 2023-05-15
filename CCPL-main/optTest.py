import torch
state_dict = torch.load("artistic/decoder_iter_160000.pth.tar")#xxx.pth或者xxx.pt就是你想改掉的权重文件
torch.save(state_dict, "art_decoder.pth", _use_new_zipfile_serialization=False)
