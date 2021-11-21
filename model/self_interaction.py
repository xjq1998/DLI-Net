import torch


class SelfInteraction:
    def __init__(self,k):
        self.k=k;

    def __call__(self, fea):
        b, n, f = fea.shape
        attn = fea @ fea.transpose(-2, -1)
        # [b,n,n]
        keep_num = int(n * self.k)
        keep_index = torch.topk(torch.sum(attn, -1), keep_num, -1, largest=False)[1]
        # [b,keep_num]
        keep_index,_=torch.sort(keep_index,dim=1)
        sparse_fea = torch.gather(
            fea, 1, keep_index.unsqueeze(2).expand(b, keep_num, f)
        )
        # [B,keep_num,C]
        return sparse_fea

    def visualize(self, fea):
        b, n, f = fea.shape
        attn = fea @ fea.transpose(-2, -1)
        # [b,n,n]
        keep_num = int(n * self.k)
        keep_index = torch.topk(torch.sum(attn, -1), keep_num, -1, largest=False)[1]
        # [b,keep_num]
        sparse_fea = torch.gather(
            fea, 1, keep_index.unsqueeze(2).expand(b, keep_num, f)
        )
        # [B,keep_num,C]
        out=(sparse_fea,keep_index)
        return out