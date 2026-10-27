"""JEV两个独立的全参计算图，不缓存冻结特征，不传播跨模型联合梯度。"""
import torch


class GateGraph(torch.nn.Module):
    def __init__(self,state,config):
        super().__init__();self.model=state.model;self.head=state.head;self.state=state;self.config=config
        self.last_probabilities=[]

    def forward(self,row):
        from .gate import encode,core
        encoded=encode(self.state,row,self.config)
        hidden=core(self.model.llm)(inputs_embeds=encoded['embeds'].unsqueeze(0),use_cache=False,return_dict=True).last_hidden_state[0,encoded['positions']]
        logits=self.head(hidden.to(self.head.weight.dtype)).float()
        self.last_probabilities=logits.detach().softmax(-1)[:,1].tolist()
        return torch.nn.functional.cross_entropy(logits,torch.tensor(encoded['labels'],device=logits.device))


class BackendGraph(torch.nn.Module):
    def __init__(self,session,config):
        super().__init__();self.model=session.model;self.session=session;self.config=config

    def forward(self,row):
        from .training import backend_example
        return self.model(**backend_example(self.session,row,self.config),use_cache=False,return_dict=True).loss
