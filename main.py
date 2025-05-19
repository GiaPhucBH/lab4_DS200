from trainer import TrainingConfig, SparkConfig, Trainer
from models import DeepImage, DeepImageCNN, DeepImageGradientBoosting
from transforms import Transforms, RandomHorizontalFlip, Normalize


transforms = Transforms([
    RandomHorizontalFlip(p=0.345),
    Normalize(
        mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
        std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)
    )
])

if __name__ == "__main__":
   
    train_config = TrainingConfig(
        num_samples=9e4,  
        batch_size=128,  
        max_epochs=100,
        learning_rate=3e-5,
        alpha=5e-4,
        model_name="DeepImageCNN",  
        ckpt_interval=1,
        ckpt_interval_batch=90000 // 128,  
        load_model="epoch-5",
        verbose=True
    )

    
    spark_config = SparkConfig(
        appName="CINIC",
        receivers=4,
        host="local",
        stream_host="localhost",
        port=6100,
        batch_interval=4,
        split="train"  
    )

    
    deep_feature = DeepImage(num_classes=10, modelName="ResNet50")
    deep_cnn = DeepImageCNN(num_classes=10)
    deep_gb = DeepImageGradientBoosting(num_classes=10)

    
    print("Generating features with DeepImage...")
    trainer_feature = Trainer(deep_feature, "train", train_config, spark_config, transforms)
    trainer_feature.train()

    
    print("Training DeepImageCNN...")
    train_config.model_name = "DeepImageCNN"
    trainer_cnn = Trainer(deep_cnn, "train", train_config, spark_config, transforms)
    trainer_cnn.train()

    
    print("Training DeepImageGradientBoosting...")
    train_config.model_name = "DeepImageGradientBoosting"
    trainer_gb = Trainer(deep_gb, "train", train_config, spark_config, transforms)
    trainer_gb.train()

    
    print("Evaluating on valid set...")
    spark_config.split = "valid"
    train_config.model_name = "DeepImageCNN"
    trainer_cnn = Trainer(deep_cnn, "valid", train_config, spark_config, transforms)
    trainer_cnn.predict()

    train_config.model_name = "DeepImageGradientBoosting"
    trainer_gb = Trainer(deep_gb, "valid", train_config, spark_config, transforms)
    trainer_gb.predict()