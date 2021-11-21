import torch


def cal_triplet_loss(dis, margin):
        b=dis.shape[0]
        # [b,b]

        eye = torch.eye(b).cuda()
        pos_dis = torch.masked_select(dis, eye.bool())
        # [b]
        loss = pos_dis.unsqueeze(1) - dis + margin
        loss = loss * (1 - eye)
        loss = torch.nn.functional.relu(loss)

        hard_triplets = loss > 0
        # [b,b]
        num_pos_triplets = torch.sum(hard_triplets, dim=1)
        # [b]
        loss = torch.sum(loss, dim=1) / (num_pos_triplets + 1e-16)
        loss = torch.mean(loss)
        return loss