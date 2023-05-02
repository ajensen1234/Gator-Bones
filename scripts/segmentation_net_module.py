
import sys, os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import wandb
from monai.inferers import sliding_window_inference
"""
Class SegmentationNetModule

Inputs:
config: a Configuration object that contains various parameters for a segmentation trial
wandb_run: a wandb Run object that represents the current run
learning_rate: a float that represents the learning rate for the optimizer

Outputs:
None

Rationale:
This class inherits from pl.LightningModule and defines the segmentation model, 
the loss function and the training and validation steps.
It helps to train and evaluate the segmentation model using pytorch-lightning and wandb.

Future:
This class can be extended to include more metrics or callbacks for monitoring the performance.
It can also be modified to support different models or loss functions.
"""
class SegmentationNetModule(pl.LightningModule):
    	"""
    	__init__

    	Inputs:
    	config: a Configuration object that contains various parameters for a segmentation trial
    	wandb_run: a wandb Run object that represents the current run
    	learning_rate: a float that represents the learning rate for the optimizer

    	Outputs:
    	None

    	Rationale:
    	This method initializes the SegmentationNetModule object with the given config, wandb_run and learning_rate.
    	It also imports and creates the segmentation model from the ModelManager class and sets the device and loss function.
    	Future:
    	This method can be extended to take more arguments or options for initializing the segmentation model. It could include additional
	options for more complicated tasks such as multi class segmentation, or multiple instance segmentation.
    	"""
	def __init__(self, config, wandb_run, learning_rate=1e-3):
    	#def __init__(self, pose_hrnet, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters("learning_rate")
        self.config = config

        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
        print(sys.path)
	    #import models relative path
        from ModelManager import ModelManager
        self.model_manager = ModelManager(config)

        self.seg_net = self.model_manager.get_segmentor()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.pose_hrnet = pose_hrnet
        print("Type of net selected: " + self.config.model['HEAD'])
        print("Net is on device " + str(next(self.seg_net.parameters()).get_device()))     # testing line
        print("Is this net on GPU? " + str(next(self.seg_net.parameters()).is_cuda))            # testing line
        self.seg_net.to(device=device, dtype=torch.float32)                          # added recently and may fix a lot
        # *** IF the above line causes an error because you do not have CUDA, then just comment it out and the model should run, albeit on the CPU ***
        print("Net is on device " + str(next(self.seg_net.parameters()).get_device()))     # testing line
        print("Net on GPU? " + str(next(self.seg_net.parameters()).is_cuda))            # testing line

        self.wandb_run = wandb_run
        self.loss_fn = self.config.model['LOSS']
        # self.loss_fn = monai.losses.DiceLoss(sigmoid=True)
        #print(self.pose_hrnet.get_device())

   
    def forward(self, x):
        """This performs a forward pass on the dataset

        Args:
            x (this_type): This is a tensor containing the information yaya

        Returns:
            the forward pass of the dataset: using a certain type of input
        """
        return self.seg_net(x)

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters, lr=1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    """
    training_step

    Inputs:
    train_batch: a dictionary that contains a batch of input images and labels for training
    batch_idx: an integer that represents the index of the current batch

    Outputs:
    loss: a torch.Tensor that represents the loss value for the current batch

    Rationale:
    This method performs a training step on the current batch using the segmentation model and the loss function loaded from config. 
    It also logs the loss value to wandb.
    Future:
    This method can be extended to include more metrics or callbacks for monitoring the training performance. 
    It can also be modified to use different optimizers or learning rate schedulers.
    """
    def training_step(self, train_batch, batch_idx):
        training_batch, training_batch_labels = train_batch['image'], train_batch['label']
        x = training_batch
        #print("Training batch is on device " + str(x.get_device()))         # testing line
        training_output = self.seg_net(x)
        loss = self.loss_fn(training_output, training_batch_labels)
        #self.log('exp_train/loss', loss, on_step=True)
        #self.wandb_run.log('train/loss', loss, on_step=True)
        self.wandb_run.log({'train/loss': loss})
        #self.log(name="train/loss", value=loss)
        return loss
    """
    validation_step

    Inputs:
    validation_batch: a dictionary that contains a batch of input images and labels for validation
    batch_idx: an integer that represents the index of the current batch

    Outputs:
    loss: a torch.Tensor that represents the loss value for the current batch

    Rationale:
    This method performs a validation step on the current batch using the segmentation model and the loss function. It also logs the loss value and an image of the validation output to wandb. 
    It uses sliding window inference to handle large input images. Further research might be needed for use case of sliding window inference.
    In monai tutorials for swinUNETR they use sliding window inference, and it doesn't seem to worsen performance.
    Future:
    This method can be extended to include more metrics or callbacks for monitoring the validation performance.
    It can also be modified to use different inference methods or parameters.
    """
    def validation_step(self, validation_batch, batch_idx):
        val_batch, val_batch_labels = validation_batch['image'], validation_batch['label']
        x = val_batch
       	#print("Validation batch is on device " + str(x.get_device()))       # testing line
        #val_output = self.seg_net(x)
        roi_size = (512, 512)
        sw_batch_size = 4
        val_output = sliding_window_inference(x, roi_size, sw_batch_size, self)
        loss = self.loss_fn(val_output, val_batch_labels)
        #self.log('validation/loss', loss)
        #self.wandb_run.log('validation/loss', loss, on_step=True)
        self.wandb_run.log({'validation/loss': loss})
        #self.log('validation/loss', loss)
        image = wandb.Image(val_output, caption='Validation output')
        self.wandb_run.log({'val_output': image})
        return loss
    """
    test_step

    Inputs:
    test_batch: a dictionary that contains a batch of input images and labels for testing
    batch_idx: an integer that represents the index of the current batch

    Outputs:
    loss: a torch.Tensor that represents the loss value for the current batch

    Rationale:
    This method performs a test step on the current batch using the segmentation model and the loss function. 
    It also logs the loss value to wandb.
    Future:
    This method can be extended to include more metrics or callbacks for monitoring the test performance. 
    It can also be modified to save or visualize the test output.
    """
    def test_step(self, test_batch, batch_idx):
        test_batch, test_batch_labels = test_batch['image'], test_batch['label']
        x = test_batch
        test_output = self.seg_net(x)
        loss = self.loss_fn(test_output, test_batch_labels)
        #self.log('test/loss', loss)
        #self.wandb_run.log('test/loss', loss, on_step=True)
        #self.wandb_run.log({'test/loss': loss})
        #self.on_test_batch_end(self, outputs=test_output, batch=test_batch, batch_idx=batch_idx)
        #self.on_test_batch_end(outputs=test_output, batch=test_batch, batch_idx=batch_idx, dataloader_idx=0)
        return loss
    
