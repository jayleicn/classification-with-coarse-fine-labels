local Criterion_jielei = torch.class('nn.Criterion_jielei')

function Criterion_jielei:__init()
   self.gradInput = torch.Tensor()
   self.output = 0
end

function Criterion_jielei:updateOutput(input, target, indicator)
end

function Criterion_jielei:forward(input, target, indicator)
   return self:updateOutput(input, target, indicator)
end

function Criterion_jielei:backward(input, target, indicator)
   return self:updateGradInput(input, target, indicator)
end

function Criterion_jielei:updateGradInput(input, target, indicator)
end

function Criterion_jielei:clone()
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   return clone
end

function Criterion_jielei:type(type, tensorCache)
   assert(type, 'Criterion: must provide a type to convert to')
   -- find all tensors and convert them
   for key,param in pairs(self) do
      self[key] = nn.utils.recursiveType(param, type, tensorCache)
   end
   return self
end

function Criterion_jielei:float()
   return self:type('torch.FloatTensor')
end

function Criterion_jielei:double()
   return self:type('torch.DoubleTensor')
end

function Criterion_jielei:cuda()
   return self:type('torch.CudaTensor')
end

function Criterion_jielei:__call__(input, target, indicator)
   self.output = self:forward(input, target, indicator)
   self.gradInput = self:backward(input, target, indicator)
   return self.output, self.gradInput
end
