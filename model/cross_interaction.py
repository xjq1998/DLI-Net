import torch


class CrossInteraction:
    def __init__(self, norm_type):
        self.norm_type = norm_type

    def __call__(self, img_fea, sk_fea):
        bi, n, _ = img_fea.shape
        bs = sk_fea.shape[0]
        img_fea = img_fea.permute(2, 0, 1).flatten(1)
        # [f,bi*n]
        img_fea = img_fea.permute(1, 0).unsqueeze(0)
        # [1,bi*n,f]
        sk_fea = sk_fea.permute(2, 0, 1).flatten(1)
        # [f,bs*n]
        sk_fea = sk_fea.permute(1, 0).unsqueeze(0)
        # [1,bs*n,f]
        local_dis = torch.cdist(sk_fea, img_fea)[0]
        # [bs*n,bi*n]
        local_dis = local_dis.reshape([bs, n, bi, n])
        # [bs,n,bi,n]
        local_dis = torch.min(local_dis, dim=3)[0]
        # [bs,n,bi]
        local_dis = local_dis.permute(1, 0, 2)
        # [n,bs,bi]
        if self.norm_type == "2norm":
            dis = torch.norm(local_dis, p=2, dim=0)
        elif self.norm_type == "1norm":
            dis = torch.sum(local_dis, dim=0)
        return dis

    def visualize(self, img_fea, sk_fea):
        bi, n, _ = img_fea.shape
        bs = sk_fea.shape[0]
        img_fea = img_fea.permute(2, 0, 1).flatten(1)
        # [f,bi*n]
        img_fea = img_fea.permute(1, 0).unsqueeze(0)
        # [1,bi*n,f]
        sk_fea = sk_fea.permute(2, 0, 1).flatten(1)
        # [f,bs*n]
        sk_fea = sk_fea.permute(1, 0).unsqueeze(0)
        # [1,bs*n,f]
        local_dis = torch.cdist(sk_fea, img_fea)[0]
        # [bs*n,bi*n]
        local_dis = local_dis.reshape([bs, n, bi, n])
        # [bs,n,bi,n]
        local_dis, match_index = torch.min(local_dis, dim=3)
        # [bs,n,bi]
        local_dis = local_dis.permute(1, 0, 2)
        # [n,bs,bi]
        match_index = match_index.permute(1, 0, 2)
        if self.norm_type == "2norm":
            dis = torch.norm(local_dis, p=2, dim=0)
        elif self.norm_type == "1norm":
            dis = torch.sum(local_dis, dim=0)
        return (local_dis, match_index)
