import torch

checkpoint = torch.load("pretrainmodel/deit_base_patch16_224-b5f2ef4d.pth", map_location='cpu')


reduce_neck = []
with open("importance/kl5k/Deit_base_12_neck_768_kl_5k_192.txt", 'r') as f:
    for i in f:
        reduce_neck.append(int(i))

new_dict  = {}
cnt = 1
for k, v in checkpoint['model'].items():
    print(k,end= ", ")
    print(v.shape)
    new_dict[ k] = v

for k, v in checkpoint['model'].items():
    print(k,end= ", ")
    if "qkv.weight" in k or "head.weight" in k or "mlp.fc1.weight" in k:
        new_index = [i not in  reduce_neck for i in torch.arange(v.size(1))]
        new_v = v[:,new_index]
        print(new_v.shape)
        new_dict[ k] = new_v
    elif "cls_token" in k or "pos_embed" in k:
        new_index = [i not in  reduce_neck for i in torch.arange(v.size(2))]
        new_v = v[:,:,new_index]
        print(new_v.shape)
        new_dict[ k] = new_v
    elif "patch_embed" in k or "norm" in k  or "fc2" in k or "attn.proj" in k:
        new_index = [i not in  reduce_neck for i in torch.arange(v.size(0))]
        new_v = v[new_index]
        print(new_v.shape)
        new_dict[ k] = new_v
    else:
        print(v.shape)
        new_dict[ k] = v



checkpoint = new_dict

reduce_attn = []
for i in range(12):
    reduce_i = []
    file_name = "importance/kl5k/Deit_base_12_attn_768_kl_" + str(i) + "_5k_importance_rank_multihead3.txt"
    with open(file_name, 'r') as f: ## 
        for t in f:
            reduce_i.append(int(t))
    reduce_attn.append(reduce_i)

reduce_mlp = []
for i in range(12):
    reduce_i = []
    file_name = "importance/kl5k/Deit_base_12_ffn_3072_kl_" + str(i) + "_5k_importance_rank_768.txt"
    with open(file_name, 'r') as f: ## 
        for t in f:
            reduce_i.append(int(t))
    reduce_mlp.append(reduce_i)


for reduce in range(0,12):
    block_ind = "blocks." + str(reduce) + ".attn.qkv.weight"
    print(block_ind,end= ", ")
    v = checkpoint[block_ind]
    interval = v.size(0) // 3
    new_index = [i % interval not in  reduce_attn[reduce] for i in torch.arange(v.size(0))]
    new_v = v[new_index]
    print(new_v.shape)
    new_dict[block_ind] = new_v       
    block_ind = "blocks." + str(reduce) + ".attn.qkv.bias"
    print(block_ind,end= ", ")
    v = checkpoint[block_ind]
    interval = v.size(0) // 3
    new_index = [i % interval not in  reduce_attn[reduce] for i in torch.arange(v.size(0))]
    new_v = v[new_index]
    new_dict[block_ind] = new_v       
    print(new_v.shape)
    block_ind = "blocks." + str(reduce) + ".attn.proj.weight"
    print(block_ind,end= ", ")
    v = checkpoint[block_ind]
    new_index = [i  not in  reduce_attn[reduce] for i in torch.arange(v.size(1))]
    new_v = v[:,new_index]
    new_dict[block_ind] = new_v       
    print(new_v.shape)


for reduce in range(0,12):                                                ##
    block_ind = "blocks." + str(reduce) + ".mlp.fc1.weight"
    print(block_ind,end= ", ")
    v = checkpoint[block_ind]
    new_index = [i  not in  reduce_mlp[reduce] for i in torch.arange(v.size(0))]
    new_v = v[new_index]
    new_dict[block_ind] = new_v       
    print(new_v.shape)
    block_ind = "blocks." + str(reduce) + ".mlp.fc1.bias"
    print(block_ind,end= ", ")
    v = checkpoint[block_ind]
    new_index = [i  not in  reduce_mlp[reduce] for i in torch.arange(v.size(0))]
    new_v = v[new_index]
    new_dict[block_ind] = new_v       
    print(new_v.shape)
    block_ind = "blocks." + str(reduce) + ".mlp.fc2.weight"
    print(block_ind,end= ", ")
    v = checkpoint[block_ind]
    new_index = [i  not in  reduce_mlp[reduce] for i in torch.arange(v.size(1))]
    new_v = v[:,new_index]
    new_dict[block_ind] = new_v       
    print(new_v.shape)

torch.save(new_dict, "up_deit_t_5k_init.pth")    ##