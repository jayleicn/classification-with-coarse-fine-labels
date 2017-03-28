testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local batchNumber
local top1_center_fine, top1_center_coarse, loss
local timer = torch.Timer()
local nTest = 10000

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_center_fine = 0
   top1_center_coarse = 0
   loss = 0
   for i=1,nTest/opt.batchSize do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = math.min((indexStart + opt.batchSize - 1), nTest)
      local inputs = testData.data[{{indexStart, indexEnd}, {}, {}, {}}]
      local labelsFine = testData.labelFine[{{indexStart, indexEnd}}]
      local labelsCoarse = testData.labelCoarse[{{indexStart, indexEnd}}]
      local labels = torch.cat(labelsFine:view(-1,1), labelsCoarse:view(-1,1), 2)
      local indicators = testData.indicator[{{indexStart, indexEnd},{}}]
      testBatch(inputs, labels, indicators)
   end

   top1_center_fine = top1_center_fine * 100 / nTest
   top1_center_coarse = top1_center_coarse * 100 / nTest
   loss = loss / torch.ceil(nTest/opt.batchSize) -- because loss is calculated per batch
   testLogger:add{
      ['avg loss (test set)'] = loss,
      ['% top1_fine accuracy (test set) (center crop)'] = top1_center_fine,
      ['% top1_coarse accuracy (test set) (center crop)'] = top1_center_coarse
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1_fine %.2f\t '
                          .. 'accuracy [Center](%%):\t top-1_coarse %.2f\t ',
                       epoch, timer:time().real, loss, top1_center_fine, top1_center_coarse))

   print('\n')


end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()
local indicators = torch.CudaTensor()

function testBatch(inputsCPU, labelsCPU, indicatorsCPU)
   batchNumber = batchNumber + opt.batchSize

   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)
   indicators:resize(indicatorsCPU:size()):copy(indicatorsCPU)

   -- local outputs = model:forward(inputs)
   -- local err = criterion:forward(outputs, labels)  -- they are tensors, should be two tensor inside a table
   local err = 0
   local outputs = model:forward(inputs)  --  {batchSize * #fineClass , batchSize * #coarseClass}
   for i=1, inputs:size()[1] do   ---inputs: nxcxhxw
      err_tmp = criterion:forward( {outputs[1][i], outputs[2][i]}, labels[i], indicators[i])
      err = err + err_tmp  -- add over the mini-batch
   end  
   
   
   cutorch.synchronize()
   local pred_fine = outputs[1]:float()
   local pred_coarse = outputs[2]:float()

   loss = loss + ( err / inputs:size()[1] )

   local _, pred_fine_sorted = pred_fine:sort(2, true)
   local _, pred_coarse_sorted = pred_coarse:sort(2, true)
   for i=1,pred_fine:size(1) do
      local g = labelsCPU[i][1]  --keep in mind this is a tensor, not a table
      local h = labelsCPU[i][2]
      if pred_fine_sorted[i][1] == g then top1_center_fine = top1_center_fine + 1 end
      if pred_coarse_sorted[i][1] == h then top1_center_coarse = top1_center_coarse + 1 end
   end
   
   
   if batchNumber % 1024 == 0 then
      print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, nTest))
   end
end
