class DefaultModel:
    backbone_type = "roberta-base"
    pretrained_backbone = True
    backbone_hidden_dropout =  0.
    backbone_hidden_dropout_prob =  0.
    backbone_attention_dropout =  0.
    backbone_attention_probs_dropout_prob = 0.
    pooling_type = 'MeanPooling'
    freeze_n_layers = 0
    reinitialize_n_layers = 0

class SchedulerConfig:
    scheduler_type = "cosine_schedule_with_warmup"
    batch_scheduler: True

    # constant_schedule_with_warmup, linear_schedule_with_warmup:
    n_warmup_steps =  0
    
    # cosine_schedule_with_warmup:
    n_cycles= 0.5

    # polynomial_decay_schedule_with_warmup:
    power: 1.0
    min_lr: 0.0

class DefaultConfig:
    use_wandb = False
    model = DefaultModel
    all_special_tokens = []
    tokenizer_dir_path = "./tokenizer"
    tokenizer = None
    
    # training 
    train_batch_size = 32
    epochs = 10
    apex=  True
    unscale = False
    max_grad_norm = 1000
    train_print_frequency = 50

    # awp 
    adversarial_lr: 0.0001
    adversarial_eps: 0.005
    adversarial_epoch_start: 99
    
    # data 
    n_workers = 12
    token_max_length = 1024

    # optimizer
    encoder_lr = 0.00001
    embeddings_lr = 0.00001
    decoder_lr = 0.00001
    
    criterion_type = "BCELoss"
    class_weigts = None

    # scheduler
    schedulr = SchedulerConfig