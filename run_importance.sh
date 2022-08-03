#! /bin/sh
CUDA_VISIBLE_DEVICES=0 python test_neck_importance_score.py

CUDA_VISIBLE_DEVICES=0 python test_attn_importance_score.py --block_ind 0
CUDA_VISIBLE_DEVICES=0 python test_attn_importance_score.py --block_ind 1
CUDA_VISIBLE_DEVICES=0 python test_attn_importance_score.py --block_ind 2
CUDA_VISIBLE_DEVICES=0 python test_attn_importance_score.py --block_ind 3
CUDA_VISIBLE_DEVICES=0 python test_attn_importance_score.py --block_ind 4
CUDA_VISIBLE_DEVICES=0 python test_attn_importance_score.py --block_ind 5
CUDA_VISIBLE_DEVICES=0 python test_attn_importance_score.py --block_ind 6
CUDA_VISIBLE_DEVICES=0 python test_attn_importance_score.py --block_ind 7
CUDA_VISIBLE_DEVICES=0 python test_attn_importance_score.py --block_ind 8
CUDA_VISIBLE_DEVICES=0 python test_attn_importance_score.py --block_ind 9
CUDA_VISIBLE_DEVICES=0 python test_attn_importance_score.py --block_ind 10
CUDA_VISIBLE_DEVICES=0 python test_attn_importance_score.py --block_ind 11

CUDA_VISIBLE_DEVICES=0 python test_ffn_importance_score.py --reduce 0
CUDA_VISIBLE_DEVICES=0 python test_ffn_importance_score.py --reduce 1
CUDA_VISIBLE_DEVICES=0 python test_ffn_importance_score.py --reduce 2
CUDA_VISIBLE_DEVICES=0 python test_ffn_importance_score.py --reduce 3
CUDA_VISIBLE_DEVICES=0 python test_ffn_importance_score.py --reduce 4
CUDA_VISIBLE_DEVICES=0 python test_ffn_importance_score.py --reduce 5
CUDA_VISIBLE_DEVICES=0 python test_ffn_importance_score.py --reduce 6
CUDA_VISIBLE_DEVICES=0 python test_ffn_importance_score.py --reduce 7
CUDA_VISIBLE_DEVICES=0 python test_ffn_importance_score.py --reduce 8
CUDA_VISIBLE_DEVICES=0 python test_ffn_importance_score.py --reduce 9
CUDA_VISIBLE_DEVICES=0 python test_ffn_importance_score.py --reduce 10
CUDA_VISIBLE_DEVICES=0 python test_ffn_importance_score.py --reduce 11