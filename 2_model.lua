-- 2_model.lua
require 'nn'
require 'cunn'

if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain)
   model = torch.load(opt.retrain)
else
   paths.dofile('models/' .. opt.netType .. '.lua')
   print('=> Creating model from file: models/' .. opt.netType .. '.lua')
   model = createModel() -- for the model creation code, check the models/ folder
--   paths.dofile('vgg_m_jie.lua')  --model here
   if opt.backend == 'cudnn' then
      require 'cudnn'
      cudnn.convert(model, cudnn)
   elseif opt.backend == 'cunn' then
      require 'cunn'
      model = model:cuda()
   elseif opt.backend ~= 'nn' then
      error'Unsupported backend'
   end
end

-- 3_loss.lua
require 'Criterion_jielei.lua'
require 'ParallelCriterion_jielei.lua'

criterion100 = nn.ClassNLLCriterion()
criterion20 = nn.ClassNLLCriterion()

criterion = nn.ParallelCriterion_jielei():add(criterion100):add(criterion20)

-- print(model)

print('=> Criterion')
print(criterion)

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
-- model is converted to CUDA in the init script itself
-- model = model:cuda()
criterion:cuda()

collectgarbage()
