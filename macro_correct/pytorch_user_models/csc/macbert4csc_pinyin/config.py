# 模型配置
model_config = {
    "pretrained_model_name_or_path": "hfl/chinese-macbert-base",  # 基础MacBERT
    "model_save_path": "output/macbert4csc_pinyin",
    
    # 训练参数
    "batch_size": 32,
    "max_len": 128,
    "epochs": 10,
    "lr": 2e-5,
    "loss_det_rate": 0.3,
    
    # 拼音融合参数
    "pinyin_embed_dim": 128,
    "fusion_type": "gate",  # gate, attention, add
    
    # 数据路径
    "path_train": "macro_correct/corpus/text_correction/sighan/sighan2015.train.json",
    "path_dev": "macro_correct/corpus/text_correction/sighan/sighan2015.dev.json",
}