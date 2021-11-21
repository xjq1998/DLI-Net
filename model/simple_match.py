import torch


class SimpleMatch:
    def __init__(self, norm_type):
        self.norm_type = norm_type

    def __call__(self, img_fea, sk_fea):
        if len(img_fea.shape) == 2:
            dis = torch.cdist(sk_fea.unsqueeze(0), img_fea.unsqueeze(0))[0]
        else:
            img_fea = img_fea.permute(1, 0, 2)
            # [n,bi,f]
            sk_fea = sk_fea.permute(1, 0, 2)
            # [n,bs,f]
            local_dis = torch.cdist(sk_fea.contiguous(), img_fea.contiguous())
            # [n,bs,bi]
            if self.norm_type == "2norm":
                dis = torch.norm(local_dis, p=2, dim=0)
            elif self.norm_type == "1norm":
                dis = torch.sum(local_dis, dim=0)
        return dis
