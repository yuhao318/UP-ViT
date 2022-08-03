import os
import re

floder = "importance/kl5k/importance/"
reduce = 0
def rank_neck():
    f1 = open(floder + "Deit_base_12_neck_768_kl_5k.txt", 'r')
    kl = []
    for i in f1:
        top1 = float(re.findall(r"\d+\.?\d*",i)[-1])
        kl.append(top1)

    print(kl)

    sorted_id = sorted(range(len(kl)), key=lambda k: kl[k])
    print(sorted_id)

    with open("importance/kl5k/" + "Deit_base_12_neck_768_kl_5k_192.txt", 'w') as f:
        for s in sorted_id[:576]:
            f.write(str(s) + '\n')
rank_neck()



floder = "importance/kl5k/importance/"
reduce = 0
def rank_ffn():
    for ind in range(12):
        f1 = open(floder + "Deit_base_12_ffn_3072_kl_" + str(ind) + "_5k.txt", 'r')
        kl = []
        for i in f1:
            top1 = float(re.findall(r"\d+\.?\d*",i)[-1])
            kl.append(top1)

        print(kl)

        sorted_id = sorted(range(len(kl)), key=lambda k: kl[k])
        print(sorted_id)

        with open("importance/kl5k/" + "Deit_base_12_ffn_3072_kl_" + str(ind) + "_5k_importance_rank_768.txt", 'w') as f:
            for s in sorted_id[:2304]:
                f.write(str(s) + '\n')
rank_ffn()

floder =  "importance/kl5k/importance/"
def rank_attn():
    for ind in range(12):

        f1 = open(floder+ "Deit_base_12_attn_768_kl_" + str(ind) + "_5k.txt", 'r')
        kl = []
        for i in f1:
            top1 = float(re.findall(r"\d+\.?\d*",i)[-1])
            kl.append(top1)
        print(kl)
        length = len(kl)
        single = length // 3
        final_result = []
        for i in range(3):
            kl_i = kl[i*single: (i+1)*single]

            print(kl_i)

            sorted_id = sorted(range(len(kl_i)), key=lambda k: kl_i[k])
            print(sorted_id)
            final_result += [ t + i*single for t in sorted_id[:192]]
        with open("importance/kl5k/"+  "Deit_base_12_attn_768_kl_" + str(ind) + "_5k_importance_rank_multihead3.txt", 'w') as f:
            for s in final_result:
                f.write(str(s) + '\n')
rank_attn()