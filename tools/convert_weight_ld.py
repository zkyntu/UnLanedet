import torch
import pickle as pkl
import sys

if __name__=="__main__":
    # example
    # python tools/convert_weight_ld.py old_ckpt_path new_ckpt_path
    path = sys.argv[1]
    ckpt = torch.load(path,map_location='cpu')['model']
    
    newmodel = {}
    
    for k in list(ckpt.keys()):
        old_key = k
        k = 'backbone.' + k
        print(f"{old_key} --> {k}")
        newmodel[k] = ckpt.pop(old_key)
        
    res = {"model": newmodel}

    torch.save(res,sys.argv[2])
