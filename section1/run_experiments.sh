python train.py +fp16=false +scaling=no +wandb.name=final-fp32-training device="cuda:1"
python train.py +fp16=true +scaling=no +wandb.name=final-no-scaling device="cuda:1"
python train.py +fp16=true +scaling=torch +wandb.name=final-torch-scaling device="cuda:1"
python train.py +fp16=true +scaling=dynamic +wandb.name=final-dynamic-scaling device="cuda:1"
python train.py +fp16=true +scaling=static +wandb.name=final-static-scaling device="cuda:1"