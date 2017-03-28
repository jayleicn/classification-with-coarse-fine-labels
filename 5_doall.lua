require 'torch'
require 'nn'

require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('0_opts.lua')
opt = opts.parse(arg)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('1_data.lua')
paths.dofile('2_model.lua')
paths.dofile('3_train.lua')
paths.dofile('4_test.lua')

epoch = opt.epochNumber  -- used for restarts

for i=1,opt.nEpochs do   -- total number of epochs
   train()
   test()
   epoch = epoch + 1
end