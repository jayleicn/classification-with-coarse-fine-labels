-- ref: https://github.com/karim-ahmed/nofe-caffe/blob/master/nofe_examples/nofe_VGG-C100/generalist/generalist_train_val.prototxt
-- ECCV16 "Network of Experts for Large-Scale Image Categorization" by K. Ahmed et.al

require 'nn'
local utils = paths.dofile'utils.lua'

function createModel()
    local features = nn.Sequential()
    -- Stack 1 [2X conv3X3,64 ]
    features:add(nn.SpatialConvolution(3,64,3,3,1,1,1,1))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
    
    -- Stack 2 [2 layers:  conv3X3,128 ]
    features:add(nn.SpatialConvolution(64,128,3,3,1,1,1,1))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialConvolution(128,128,3,3,1,1,1,1))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialMaxPooling(2,2,2,2):ceil())    

    -- Stack 3 [4 layers:  conv3X3, 256]
    features:add(nn.SpatialConvolution(128,256,3,3,1,1,1,1))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialMaxPooling(2,2,2,2):ceil()) 

    -- FC Layers
    local classifier = nn.Sequential()
    classifier:add(nn.View(256*4*4))
    classifier:add(nn.Linear(256*4*4, 1024))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(1024, 1024))    
    classifier:add(nn.ReLU(true))      --actually we could move Dropout layer here ...

    mlp = nn.Sequential()
    mlp_concat = nn.ConcatTable()
    for i=1,100 do
       mlp_concat:add(nn.Sequential():add(nn.Dropout(0.5)):add(nn.Linear(1024, 36)):add(nn.ReLU(true)):add(nn.Reshape(1,6,6)))
    end
    mlp:add(mlp_concat):add(nn.JoinTable(1,3))


    local fine_branch = nn.Sequential()
    fine_branch:add(nn.SpatialAveragePooling(6, 6))
    fine_branch:add(nn.Reshape(100))  -- 100x1x1 ---> 100 ---> :dim()==1
    fine_branch:add(nn.LogSoftMax())

    -- coarse branch, min-pooling should be done here
    local coarse_branch = nn.Sequential()
    coarse_branch:add(nn.SplitTable(-3))

    local coarse_branch_reshape_100 = nn.ParallelTable()
    for i=1, 100 do
        coarse_branch_reshape_100:add(nn.Reshape(1, 6, 6))
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

    -- join the 5 seperate 1x4x4 --> 5x4x4
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

    -- reshape and join --> 20x4x4 --> averagepooling --> 20x1x1 --> 20
    local coarse_branch_reshape_20 = nn.ParallelTable()
    for i=1, 20 do
        coarse_branch_reshape_20:add(nn.Reshape(1, 6, 6))
    end
    coarse_branch:add(coarse_branch_reshape_20):add(nn.JoinTable(1,3)):add(nn.SpatialAveragePooling(6,6))

    coarse_branch:add(nn.Reshape(20))
    coarse_branch:add(nn.LogSoftMax())
    
    -- Concat fine_branch and coarse_branch
    local branch = nn.ConcatTable()
    branch:add(fine_branch):add(coarse_branch)

    local model = nn.Sequential():add(features):add(classifier):add(mlp):add(branch)
    
    utils.MSRinit(model)
    --print(#model:cuda():forward(torch.CudaTensor(1,3,32,32)))

    return model:cuda()
end
