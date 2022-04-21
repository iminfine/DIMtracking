import torch
import torch.nn.functional as F
import numpy as np

def odd(f):
    return int(np.ceil(f)) // 2 * 2 + 1
 
def imcrop(I,center,size,outpatch=True):
    b,c,w,h=I.shape
    box=(int(center[0])-(size[0]-1)/2,int(center[1])-(size[1]-1)/2,size[0],size[1])
    boxInbounds=(min(h-odd(box[2]),max(0,round(box[0]))),min(w-odd(box[3]),max(0,round(box[1]))),
                 odd(box[2]),odd(box[3]))
    box=boxInbounds
    if outpatch:
        Ipatch=I[:,:,box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
        return Ipatch,box
    else:
        return box



def preprocess(image,size=None):
    device=torch.cuda.current_device()
    imageon=torch.clamp(image,min=0)
    imageoff=torch.clamp(image,max=0).abs()  
    out=torch.cat((imageon,imageoff),1)
    if size is not None:
        pad=(np.ones(4,dtype=int)*size).tolist()
        out=torch.from_numpy(np.pad(out.cpu().numpy(),((0,0),(0,0),
                              (pad[0],pad[1]),(pad[2],pad[3])),'symmetric')).to(device)
    return out
'''  
def preprocess(image):
    cuda=torch.cuda.current_device()
    n, c, h, w = image.size()
    X=torch.zeros(n,2*c,h,w,device=cuda)
    for i in range(len(image)):
        for j in range(len(image[i])):
            X[i][2*(j-1)+2]=torch.clamp(image[i][j],min=0)
            X[i][2*(j-1)+3]=torch.clamp(image[i][j],max=0).abs()        
    return X
'''
def conv2_same(Input, weight,num=1):
    padding_rows = weight.size(2)-1
    padding_cols = weight.size(3)-1  
    rows_odd = (padding_rows % 2 != 0)
    cols_odd = (padding_cols % 2 != 0)
    if rows_odd or cols_odd:
        Input = F.pad(Input, [0, int(cols_odd), 0, int(rows_odd)])
    weight=torch.flip(weight,[2, 3])
    return F.conv2d(Input, weight, padding=(padding_rows // 2, padding_cols // 2), groups=num)


def DIM_matching(X,w,iterations,epsilon2):
    cuda=torch.cuda.current_device()
    v=torch.zeros_like(w)
    Y=torch.zeros(X.shape[0],len(w),X.shape[2],X.shape[3],device=cuda)
    tem1=w.clone()
    
    for i in range(len(w)):
        v[i]=torch.max(torch.tensor(0, dtype=torch.float32,device=cuda),
        w[i]/torch.max(torch.tensor(1e-6, dtype=torch.float32,device=cuda),torch.max(w[i])))
        tem1[i]=w[i]/torch.max(torch.tensor(1e-6,dtype=torch.float32,device=cuda),torch.sum(w[i]))
    w=torch.flip(tem1,[2, 3])
    sumV=torch.sum(torch.sum(torch.sum(v,0),1),1)
    epsilon1=torch.tensor(epsilon2,dtype=torch.float32,device=cuda)/torch.max(sumV)
    for count in range(iterations):
        R=torch.zeros_like(X)
        if not torch.sum(Y)==0:
            R=conv2_same(Y,v.permute(1,0,2,3))
            R=torch.clamp(R, min=0)
        E=X/torch.max(torch.tensor(epsilon2,dtype=torch.float32,device=cuda),R)
        Input=torch.zeros_like(E)
        Input=conv2_same(E,w)
        tem2=Y.clone()
        for i in range(len(Input)):
            for j in range(len(Input[i])):
                tem2[i][j]=Input[i][j]*torch.max(epsilon1,Y[i][j])
        Y=torch.clamp(tem2, min=0)
    return Y

    
