has_test = True
deterministic = True
use_custom_worker_init = True
log_interval = 20
base_seed = 112358
# log_interval = 100 # 每100个iter记录一次wei参数

# 🔧 验证配置 - 参考HitNet实现
enable_validation = False  # 是否启用训练时验证
validation = dict(
    dataset_name="COD10K-TE",  # 验证数据集名称（使用COD10K-TE完整测试集）
    frequency=1,               # 验证频率（每N个epoch）
    metrics=["mae"],          # 验证指标
    save_best=True,           # 是否保存最佳模型
    start_epoch_ratio=0.66,    # 从总epoch数的2/3开始验证（提高训练效率）
    early_stopping=dict(
        enabled=False,        # 是否启用早停
        patience=20,          # 早停耐心值
        min_delta=1e-6       # 最小改善阈值
    )
)

__BATCHSIZE = 8
# __BATCHSIZE = 4  # 从8减小到4，降低显存占用
__NUM_EPOCHS = 150
__NUM_TR_SAMPLES = 3040 + 1000
__ITER_PER_EPOCH = __NUM_TR_SAMPLES // __BATCHSIZE  # drop_last is True
__NUM_ITERS = __NUM_EPOCHS * __ITER_PER_EPOCH

train = dict(
    batch_size=__BATCHSIZE,
    num_workers=2,
    use_amp=True,
    num_epochs=__NUM_EPOCHS,
    epoch_based=True,
    num_iters=None,
    lr=1e-4, # 这是基础学习率
    grad_acc_step=1,
    # grad_acc_step=2,  # 增加梯度累积步数，保持有效batch_size=4*2=8
    optimizer=dict(
        mode="adam",
        set_to_none=False,
        group_mode="finetune", # 关键配置：启用finetune模式
        cfg=dict(
            weight_decay=0,
            diff_factor=0.1, # 关键参数：预训练参数的学习率衰减因子
        ),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(
            num_iters=0,
            initial_coef=0.01,
            mode="linear",
        ),
        mode="step",
        cfg=dict(
            milestones=int(__NUM_ITERS * 2 / 3),
            gamma=0.1,
        ),
    ),
    bn=dict(
        freeze_status=True,
        freeze_affine=True,
        freeze_encoder=False,
    ),
    data=dict(
        shape=dict(h=384, w=384),
        names=["cod10k_tr"],
        # names=["debug_tr"],
    ),
)

test = dict(
    batch_size=__BATCHSIZE,
    num_workers=2,
    clip_range=None,
    data=dict(
        shape=dict(h=384, w=384),
        names=["camo_te", "chameleon", "cod10k_te", "nc4k"],
        # names=["chameleon" ],
    ),
)
