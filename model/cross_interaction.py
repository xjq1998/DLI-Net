import torch


class CrossInteraction:
    def __init__(self, norm_type):
        self.norm_type = norm_type

    def __call__(self, ph_fea, sk_fea):
        bi, n, _ = ph_fea.shape
        bs = sk_fea.shape[0]
        ph_fea = ph_fea.permute(2, 0, 1).flatten(1)
        # [f,bi*n]
        ph_fea = ph_fea.permute(1, 0).unsqueeze(0)
        # [1,bi*n,f]
        sk_fea = sk_fea.permute(2, 0, 1).flatten(1)
        # [f,bs*n]
        sk_fea = sk_fea.permute(1, 0).unsqueeze(0)
        # [1,bs*n,f]
        local_dis = torch.cdist(sk_fea, ph_fea)[0]
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

    def get_local_dis(self,ph_fea,sk_fea,index):
        bi, n, _ = ph_fea.shape
        bs = sk_fea.shape[0]
        ph_fea = ph_fea.permute(2, 0, 1).flatten(1)
        # [f,bi*n]
        ph_fea = ph_fea.permute(1, 0).unsqueeze(0)
        # [1,bi*n,f]
        sk_fea = sk_fea.permute(2, 0, 1).flatten(1)
        # [f,bs*n]
        sk_fea = sk_fea.permute(1, 0).unsqueeze(0)
        # [1,bs*n,f]
        local_dis = torch.cdist(sk_fea, ph_fea)[0]
        # [bs*n,bi*n]
        local_dis = local_dis.reshape([bs, n, bi, n])
        # [bs,n,bi,n]
        local_dis = torch.min(local_dis, dim=3)[0]
        # [bs,n,bi]
        local_dis = local_dis.permute(1, 0, 2)
        # [n,bs,bi]
        return local_dis[index]

    # def visualize(self, ph_fea, sk_fea):
    #     bi, n, _ = ph_fea.shape
    #     bs = sk_fea.shape[0]
    #     ph_fea = ph_fea.permute(2, 0, 1).flatten(1)
    #     # [f,bi*n]
    #     ph_fea = ph_fea.permute(1, 0).unsqueeze(0)
    #     # [1,bi*n,f]
    #     sk_fea = sk_fea.permute(2, 0, 1).flatten(1)
    #     # [f,bs*n]
    #     sk_fea = sk_fea.permute(1, 0).unsqueeze(0)
    #     # [1,bs*n,f]
    #     local_dis = torch.cdist(sk_fea, ph_fea)[0]
    #     # [bs*n,bi*n]
    #     local_dis = local_dis.reshape([bs, n, bi, n])
    #     # [bs,n,bi,n]
    #     local_dis, match_index = torch.min(local_dis, dim=3)
    #     # [bs,n,bi]
    #     local_dis = local_dis.permute(1, 0, 2)
    #     # [n,bs,bi]
    #     match_index = match_index.permute(1, 0, 2)
    #     if self.norm_type == "2norm":
    #         dis = torch.norm(local_dis, p=2, dim=0)
    #     elif self.norm_type == "1norm":
    #         dis = torch.sum(local_dis, dim=0)
    #     return (local_dis, match_index)

    def visualize(self, ph_fea, sk_fea):
        bi, n, _ = ph_fea.shape
        bs, n, _ = sk_fea.shape
        assert bi == 1
        sim = torch.einsum('imp,jnp->ijmn',[sk_fea, ph_fea])
        # [bs,n,bi,n]
        sim = sim[0,0,:,:]
        sk_local_sim, match_index = torch.max(sim, dim=1)
        # print(match_index.shape)
        ph_local_sim = torch.zeros(n).cuda()
        for i in range(n):
            ph_local_sim[match_index[i]] += sk_local_sim[i]
        return sk_local_sim, ph_local_sim

if __name__=='__main__':
    match = CrossInteraction('2norm')
    ph_fea = torch.rand(1,128,1024)
    sk_fea = torch.rand(1,128,1024)
    sk_sim, ph_sim = match.visualize(ph_fea, sk_fea)
    print(sk_sim.shape)
    print(ph_sim.shape)