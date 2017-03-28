require 'optim'
-- require 'xlua'
--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     18,   1e-2,   5e-4, },
        { 19,     29,   5e-3,   5e-4  },
        { 30,     43,   1e-3,   0 },
        { 44,     52,   5e-4,   0 },
        { 53,    1e8,   1e-4,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_fine_epoch, top1_coarse_epoch, loss_epoch  -- top1_fine_epoch is the fine label prediction for one complete epoch 
local epochSize = torch.ceil( opt.dataSize/opt.batchSize )
-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk

local function tableToOutput(dataTable, scalarTable, indicatorTable)
   local data, scalarLabels, indicators
   local quantity = #scalarTable
   assert(dataTable[1]:dim() == 3)
   data = torch.Tensor(quantity, 3, 32, 32)
   scalarLabels = torch.LongTensor(quantity,2):fill(-1111) 
   indicators = torch.IntTensor(quantity,2):fill(-1)
   
   for i=1, quantity do
      data[i]:copy(dataTable[i])
      scalarLabels[i] = torch.IntTensor(scalarTable[i])
      indicators[i] = torch.IntTensor(indicatorTable[i])
   end
   return data, scalarLabels, indicators
end


function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then                              -- True if epoch == start_point
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
   end
   batchNumber = 0

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   top1_fine_epoch = 0
   top1_coarse_epoch = 0
   loss_epoch = 0
   
   local shuffle = torch.randperm(opt.dataSize) --  20000 or 25000
   for t=1,opt.dataSize,opt.batchSize do   -- 'Number of batches per epoch'   
      -- xlua.progress(t, opt.dataSize)
      local inputsTable = {}
      local labelsTable = {}
      local indicatorsTable = {}
      for i=t, math.min(t+opt.batchSize-1, opt.dataSize) do
         local input_coarse = trainDataCoarse.data[shuffle[i]]
         local label_coarse = {trainDataCoarse.labelFine[shuffle[i]], trainDataCoarse.labelCoarse[shuffle[i]]}
         local indicator_coarse = {0, 1}
         local input_fine = trainDataFine.data[shuffle[i]]
         local label_fine = {trainDataFine.labelFine[shuffle[i]], trainDataFine.labelCoarse[shuffle[i]]}
         local indicator_fine = {1, 1}
         table.insert(inputsTable, input_coarse)
         table.insert(inputsTable, input_fine)
         table.insert(labelsTable, label_coarse)
         table.insert(labelsTable, label_fine)
         table.insert(indicatorsTable, indicator_coarse)
         table.insert(indicatorsTable, indicator_fine)
      end
      local inputs, labels, indicators = tableToOutput(inputsTable, labelsTable, indicatorsTable)
      trainBatch(inputs, labels, indicators)  
   end

   top1_fine_epoch = (top1_fine_epoch*100) / (2*opt.dataSize)
   top1_coarse_epoch = (top1_coarse_epoch*100) / (2*opt.dataSize)
   loss_epoch = loss_epoch / epochSize

   trainLogger:add{
      ['avg loss (train set)'] = loss_epoch,
      ['% top1 fine accuracy (train set)'] = top1_fine_epoch,
      ['% top1 coarse accuracy (train set)'] = top1_coarse_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 fine %.2f\t'
                          .. 'accuracy(%%):\t top-1 coarse %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1_fine_epoch, top1_coarse_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   if epoch % 5 == 0 then
      model:clearState()
      torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
      torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
   end
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()
local indicators = torch.CudaTensor()

local timer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU, indicatorsCPU) 
   collectgarbage()
   timer:reset()
   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)
   indicators:resize(indicatorsCPU:size()):copy(indicatorsCPU)
   -- 128x3x224x224, 128x2, 128x2 Tensors
   -- define update routine
   local err = 0
   local outputs = {}
   feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      
      model:zeroGradParameters() 
      outputs = model:forward(inputs)  --  {batchSize * #fineClass , batchSize * #coarseClass}
      local gradOutput = {}   -- the same size with outputs
      gradOutput = nn.utils.recursiveResizeAs(gradOutput, outputs)
      nn.utils.recursiveFill(gradOutput, 0)
      for i=1, inputs:size()[1] do   ---inputs: nxcxhxw    
          err_tmp = criterion:forward( {outputs[1][i], outputs[2][i]}, labels[i], indicators[i])
          err = err + err_tmp  -- add over the mini-batch
          df_tmp = criterion:backward( {outputs[1][i], outputs[2][i]}, labels[i], indicators[i])
          gradOutput[1][i] = df_tmp[1]
          gradOutput[2][i] = df_tmp[2]
      end
      model:backward(inputs, gradOutput)
      -- normalize gradients and f(x)
      err = err / inputs:size()[1]
      gradParameters:div(inputs:size()[1])
      
      return err, gradParameters
   end
   optim.sgd(feval, parameters, optimState)

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   
   -- top-1 error
   local top1_fine = 0
   -- local top1_coarse = 0 no need to print this out
   do
      local _,pred_fine_sorted = outputs[1]:float():sort(2, true) -- descending(true)
      local _,pred_coarse_sorted = outputs[2]:float():sort(2, true) -- descending(true)
      local dSize 
      for i=1, labelsCPU:size()[1] do
         if pred_fine_sorted[i][1] == labelsCPU[i][1] then
            top1_fine_epoch = top1_fine_epoch + 1;
            top1_fine = top1_fine + 1
         end
         if pred_coarse_sorted[i][1] == labelsCPU[i][2] then
            top1_coarse_epoch = top1_coarse_epoch + 1
         end
      end
      top1_fine = (top1_fine * 100) / labelsCPU:size()[1];
   end
   -- Calculate top-1 error, and print information
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e '):format(
          epoch, batchNumber, epochSize, timer:time().real, err, top1_fine,
          optimState.learningRate))
end
