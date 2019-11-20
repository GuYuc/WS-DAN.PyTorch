##################################
# Training Config
##################################
GPU = '1,0'
workers = 8             # number of data loading workers
epochs = 160            # number of epochs
batch_size = 48         # batch size
learning_rate = 1e-3    # learning rate

# saving directory of .ckpt models
save_dir = ''
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + '/model.ckpt'


##################################
# Model Config
##################################
image_size = 448        # size of training images
num_attentions = 32     # number of attention maps
beta = 5e-2             # param for Feature Center Loss