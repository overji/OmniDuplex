import torch


class DecisionEmbedding(torch.nn.Module):
    def __init__(self,base,ids):
        super().__init__();self.base=base;self.ids=ids
        self.rows=torch.nn.Parameter(base.weight[ids].detach().float().clone())
    @property
    def weight(self):return self.base.weight
    def forward(self,ids):
        output=self.base(ids)
        for i,token in enumerate(self.ids):
            output=torch.where((ids==token).unsqueeze(-1),self.rows[i].to(output.dtype),output)
        return output


def mean_row(weight):
    total = torch.zeros(weight.shape[1], device=weight.device, dtype=torch.float32)
    for start in range(0, len(weight), 2048):
        total += weight[start:start + 2048].float().sum(0)
    return total / len(weight)
