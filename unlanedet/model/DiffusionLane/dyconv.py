import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureResize(nn.Module):
    def __init__(self, size=(10, 25)):
        super(FeatureResize, self).__init__()
        self.size = size

    def forward(self, x):
        x = F.interpolate(x, self.size)
        return x.flatten(2)

class AddMerge(nn.Module):
    def __init__(self, learnable, embed_dims, 
                 channel_aware, initial_weights,
                 init_cfg=None):
        super(AddMerge,self).__init__()
        self.learnable=learnable
        self.embed_dims=embed_dims
        self.channel_aware=channel_aware
        self.initial_weights=initial_weights

        if(self.learnable):
            if(self.channel_aware):
                self.weights=nn.Parameter(self.initial_weights[None].repeat(self.embed_dims,1))
            else:
                self.weights=nn.Parameter(self.initial_weights)
        else:
            self.register_buffer('weights', self.initial_weights)
    
    def forward(self, query_list):
        query_list=torch.stack(query_list,dim=-1)
        if(self.channel_aware):
            weights=self.weights[None,None,...]
            merged_query=(query_list*weights).sum(-1)
        else:
            merged_query=(query_list*self.weights[None,None,None]).sum(-1)
        return merged_query
    

    
class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.HIDDEN_DIM
        self.dim_dynamic = cfg.DIM_DYNAMIC
        self.num_dynamic = cfg.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

        # self.resize = FeatureResize()
        self.cfg = cfg 

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        # import pdb;pdb.set_trace()
        # roi_features = self.resize(roi_features)
        features = roi_features.permute(1, 0, 2) 
        features = features.repeat(pro_features.shape[1]//features.shape[0],1,1)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        # features = features.mean(1,keepdim=False)
        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features
