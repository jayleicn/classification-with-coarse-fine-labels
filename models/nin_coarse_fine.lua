-- Network-in-Network
require 'nn'
local utils = paths.dofile'utils.lua'

require 'cutorch'
cutorch.setDevice(opt.GPU)

function createModel()
    local model = nn.Sequential()
    local function Block(...)
      local arg = {...}
      model:add(nn.SpatialConvolution(...))
      -- model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
      model:add(nn.ReLU(true))
      return model
    end

    Block(3,192,5,5,1,1,2,2)
    Block(192,160,1,1)
    Block(160,96,1,1)
    model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
    model:add(nn.Dropout(0.5))
    Block(96,192,5,5,1,1,2,2)
    Block(192,192,1,1)
    Block(192,192,1,1)
    model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
    model:add(nn.Dropout(0.5))
    Block(192,192,3,3,1,1,1,1)
    Block(192,192,1,1)
    Block(192,100,1,1) 
    -- 100x8x8 per category feature map 
    utils.MSRinit(model)

    -- branch 1 : fine branch
    local fine_branch = nn.Sequential()
    fine_branch:add(nn.SpatialAveragePooling(8,8))
    fine_branch:add(nn.Reshape(100))  -- 100x1x1 ---> 100 ---> :dim()==1
    fine_branch:add(nn.LogSoftMax())

    -- branch 2 : coarse branch, min-pooling should be done here
    -- part 1, split 100x8x8 into 100 seperately 1x8x8
    local coarse_branch = nn.Sequential()
    coarse_branch:add(nn.SplitTable(-3))

    local coarse_branch_reshape_100 = nn.ParallelTable()
    for i=1, 100 do
        coarse_branch_reshape_100:add(nn.Reshape(1, 8, 8))
    end
    coarse_branch:add(coarse_branch_reshape_100)

    -- group the subclasses belong to the same superclass
    local hierarchy_path = paths.concat(opt.data,'fine_coarse_label_hierarchy.t7')
    local fine_coarse_label_hierachy = torch.load(hierarchy_path)

    local coarse_branch_hierachy_table = {}
    for i=1, 20 do
        coarse_branch_hierachy_table[i] = nn.ConcatTable()
        for j=1, 5 do
            coarse_branch_hierachy_table[i]:add(nn.SelectTable(fine_coarse_label_hierachy[i][j]))
        end
    end

    local coarse_branch_hierachy = nn.ConcatTable()
    for i=1, 20 do
        coarse_branch_hierachy:add(coarse_branch_hierachy_table[i])
    end
    coarse_branch:add(coarse_branch_hierachy)

    -- join the 5 seperate 1x8x8 --> 5x8x8
    local coarse_branch_join = nn.ParallelTable()
    for i=1, 20 do
        coarse_branch_join:add(nn.JoinTable(1,3))
    end
    coarse_branch:add(coarse_branch_join)

    -- minPooling
    local coarse_branch_minpooling = nn.ParallelTable()
    for i=1, 20 do
        coarse_branch_minpooling:add(nn.Min(1,3))
    end
    coarse_branch:add(coarse_branch_minpooling)

    -- reshape and join --> 20x8x8 --> averagepooling --> 20x1x1 --> 20
    local coarse_branch_reshape_20 = nn.ParallelTable()
    for i=1, 20 do
        coarse_branch_reshape_20:add(nn.Reshape(1, 8, 8))
    end
    coarse_branch:add(coarse_branch_reshape_20):add(nn.JoinTable(1,3)):add(nn.SpatialAveragePooling(8,8))

    coarse_branch:add(nn.Reshape(20))
    coarse_branch:add(nn.LogSoftMax())
    
    -- Concat fine_branch and coarse_branch, note the two branches are in parallel
    local branch = nn.ConcatTable()
    branch:add(fine_branch):add(coarse_branch)
    model:add(branch)

    model:cuda()
    return model
end
