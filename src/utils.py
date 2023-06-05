from dataclasses import asdict
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import glob
import os 
from src.datasets import CatDataset


def initialize_train(train_config,data_config,model_config):

    preprocess = transforms.Compose(
        [
            transforms.Resize((model_config.sample_size,model_config.sample_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # TODO
        ]
    )
    train_dataset = CatDataset(data_config.training_files,
                            transform=preprocess, 
                            all_in_memory=data_config.all_in_memory)
    train_loader = DataLoader(train_dataset,
                            data_config.train_batch_size,
                            shuffle=data_config.train_shuffle)
    
    criterion = nn.MSELoss()

    noise_scheduler = DDPMScheduler(num_train_timesteps=train_config.num_train_timesteps)
    model = UNet2DModel(**asdict(model_config)).to(train_config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=train_config.lr_warmup_steps,
                num_training_steps=(len(train_loader) * train_config.epochs),
            )

    try:
        checkpoint = load_checkpoint(train_config.checkpoint_dir,train_config.device)
        model.load_state_dict(checkpoint['model'])
        # Если в конфигах другой lr, то меняет lr как в конфиге.
        initial_lr = checkpoint['optimizer']['param_groups'][0]['initial_lr']
        cur_lr = checkpoint['optimizer']['param_groups'][0]['lr']
        if initial_lr != train_config.lr:
          checkpoint['optimizer']['param_groups'][0]['initial_lr'] = train_config.lr
          checkpoint['optimizer']['param_groups'][0]['lr'] = train_config.lr
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        train_config.cur_iteration = checkpoint['iteration'] + 1
        train_config.cur_epoch = checkpoint['epoch']

        print(f"Using optimizer initial_lr={optimizer.state_dict()['param_groups'][0]['initial_lr']} lr={optimizer.state_dict()['param_groups'][0]['lr']}")

    except Exception as err: 
        print("Failed load checkpoint. Error: ", err)


    train_params = {
        'model' : model,
        'criterion' : criterion,
        'optimizer' : optimizer,
        'loader' : train_loader,
        'noise_scheduler' : noise_scheduler,
        'lr_scheduler' : lr_scheduler,
        'config' : train_config,
    }


    return train_params

def load_checkpoint(checkpoint_dir,device):

    checkpoints = list(map(lambda x : os.path.basename(x).split('_')[1].split('.')[0], glob.glob(checkpoint_dir + 'model_*')))
    if len(checkpoints) == 0:
        raise Exception("Couldn't find checkpoint")

    max_idx = 0
    for checkpoint in checkpoints:
        try:
            max_idx = max(int(checkpoint),max_idx)
        except:
            pass

    checkpoint_name = 'model_' + str(max_idx) + '.pth'
    checkpoint_dict = torch.load(checkpoint_dir + checkpoint_name,map_location=torch.device(device))
    print("Successfully loading: ", checkpoint_name)

    return checkpoint_dict


def save_checkpoint(path, model, optimizer, scheduler, epoch, iteration):

    checkpoint = { 
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler' : scheduler.state_dict(),
        'iteration' : iteration
    }
    checkpoint_name = f"model_{iteration}.pth"

    print("Saving model, optimizer and scheduler: ", checkpoint_name)
    os.makedirs(path, exist_ok=True)
    torch.save(checkpoint, path + checkpoint_name)


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def evaluate(pipeline, config):

    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        num_inference_steps=config.num_train_timesteps
    ).images

    image_grid = make_grid(images, rows=4, cols=4)
    return image_grid

def train_batch(clean_images,model,criterion,optimizer,lr_scheduler,noise_scheduler, device):

    model.train()

    clean_images = clean_images.to(device)
    noise = torch.randn(clean_images.shape).to(device)
    batch_size = clean_images.shape[0]
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
    ).long()

    noisy_images = noise_scheduler.add_noise(clean_images,noise,timesteps)
    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
    loss = criterion(noise_pred, noise)

    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    return {'loss' : loss.item()}

def train(model,criterion,optimizer,loader,noise_scheduler,lr_scheduler,config):

    global_step = config.cur_iteration

    if config.board:
        writer = SummaryWriter(log_dir=config.log_dir,flush_secs=60,max_queue=3)

    for epoch in range(config.cur_epoch,config.epochs):

        train_loss = 0

        for batch in loader:

            history = train_batch(batch,
                                model,
                                criterion,
                                optimizer,
                                lr_scheduler,
                                noise_scheduler,
                                config.device)
            

            if global_step % config.checkpoint_interval == 0:
                save_checkpoint(config.checkpoint_dir, model, optimizer, lr_scheduler, epoch, global_step)

            if config.board:
                if global_step % config.log_interval == 0:
                    writer.add_scalar('train/loss', history['loss'], global_step)
                if global_step % config.eval_interval == 0:
                    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
                    images = evaluate(pipeline, config)
                    images = np.array(images).transpose((2,0,1))
                    writer.add_image('evalute',images,global_step)

            global_step += 1
            train_loss += history['loss']

        train_loss /= len(loader)
        print(f"[{epoch + 1}/{config.epochs}][Iteration {global_step}] Loss : {train_loss}")
            

    # Сохраняем обученную модель, добавляем на доску последний лосс, генирируем изображения
    save_checkpoint(config.checkpoint_dir, model, optimizer, lr_scheduler, epoch,global_step)
    if config.board:
        writer.add_scalar('train/loss', history['loss'], global_step)
        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
        images = evaluate(pipeline, config)
        images = np.array(images).transpose((2,0,1))
        writer.add_image('evalute',images,global_step)
        writer.close()
        

        
