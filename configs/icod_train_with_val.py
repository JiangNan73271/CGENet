"""
ZoomNext训练配置 - 带验证功能
参考HitNet的验证策略，使用COD10K-TE作为验证集
"""

has_test = True  # 训练后不自动测试，只进行验证
deterministic = True
use_custom_worker_init = True
log_interval = 20
base_seed = 112358

# 🔧 验证配置 - 参考HitNet实现
enable_validation = True  # 启用训练时验证
validation = dict(
    dataset_name="cod10k_te",  # 验证数据集：COD10K-TE（完整测试集，2026张图片）
    frequency=1,               # 每个epoch进行验证
    metrics=["mae"],          # 主要评估指标：MAE（参考HitNet）
    save_best=True,           # 保存最佳模型
    min_epoch_to_save=10,     # 最小保存epoch数（参考HitNet的设置）
    early_stopping=dict(
        enabled=False,        # 是否启用早停（可根据需要开启）
        patience=20,          # 早停耐心值
        min_delta=1e-6       # 最小改善阈值
    )
)

__BATCHSIZE = 8
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
    lr=0.0001,
    grad_acc_step=1,
    optimizer=dict(
        mode="adam",
        set_to_none=False,
        group_mode="finetune",
        cfg=dict(
            weight_decay=0,
            diff_factor=0.1,
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
        # names=["debug_tr"],  # 调试时可以使用小数据集
    ),
)

# 测试配置（用于最终测试，不在训练过程中使用）
test = dict(
    batch_size=__BATCHSIZE,
    num_workers=2,
    clip_range=None,
    data=dict(
        shape=dict(h=384, w=384),
        names=["camo_te", "chameleon", "cod10k_te", "nc4k"],  # 完整测试集
    ),
)
