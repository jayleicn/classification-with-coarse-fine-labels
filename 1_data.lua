-- 1_data.lua
-- cifar100 data is preprocessed using PCA and whitening
-- and the 50000 training images are split randomly into 20000 + 20000 + 10000.
-- also add indicator item, where {1,1} for both fine and coarse, {0,1} for coarse only
-- Download whitened images form https://yadi.sk/d/em4b0FMgrnqxy, credit to @szagoruyko

local cifar_path = paths.concat(opt.data, 'cifar100_whitened.t7')
local hierarchy_path = paths.concat(opt.data,'fine_coarse_label_hierarchy.t7')
local coarse_path = paths.concat(opt.data, 'train_coarse_tmp.t7')
local fine_path = paths.concat(opt.data, 'train_fine_tmp.t7')
local val_path = paths.concat(opt.data, 'val_tmp.t7')

local files_not_exist = not paths.filep(coarse_path) and not paths.filep(fine_path) and not paths.filep(val_path)
if files_not_exist then
    -- add coarse label to the dataset
    local cifar100 = torch.load(cifar_path)
    local hierachy_maps = torch.load(hierarchy_path)

    inv_maps = {}
    for i=1,20 do
        for j=1,5 do
            inv_maps[hierachy_maps[i][j]] = i
        end
    end

    cifar100.trainData['labelFine'] = cifar100.trainData.labels
    cifar100.trainData.labels = nil
    cifar100.testData['labelFine'] = cifar100.testData.labels
    cifar100.testData.labels = nil
    cifar100.trainData['labelCoarse'] = torch.IntTensor(cifar100.trainData.size(),1):fill(0)
    cifar100.testData['labelCoarse'] = torch.IntTensor(cifar100.testData.size(),1):fill(0)
    for i=1,cifar100.trainData.size() do
        cifar100.trainData.labelCoarse[i] = inv_maps[cifar100.trainData.labelFine[i]]
    end
    for i =1,cifar100.testData.size() do
        cifar100.testData.labelCoarse[i] = inv_maps[cifar100.testData.labelFine[i]]
    end

    -- Randomly split trainData 50000 into three subsets 
    -- 1) trainCoarse: with only coarse labels 20000
    -- 2) trainFine : with both fine and coarse labels 20000
    -- 3) val : for validation 10000
    torch.manualSeed(2)
    fine_perm = torch.randperm(500)
    cifar100.trainData['indicator'] = torch.IntTensor(50000,2):fill(1)
    for i=1,100 do
       count = 0
       for j=1,50000 do
           if cifar100.trainData.labelFine[j] == i then
              count = count + 1
              if fine_perm[count] <= 250 then 
                    cifar100.trainData.indicator[j][1] = 0
              end
           end         
       end        
    end

    -- Initialize fine dataset
    train_fine_tmp = {
        data = torch.DoubleTensor(20000,3,32,32):fill(0),
        labelCoarse = torch.IntTensor(20000):fill(0),
        labelFine = torch.IntTensor(20000):fill(0),
        indicator = torch.IntTensor(20000,2):fill(1),
        size = function () return 20000 end
    } --20000

    -- Initialize coarse dataset
    train_coarse_tmp = {
        data = torch.DoubleTensor(20000,3,32,32):fill(0),
        labelCoarse = torch.IntTensor(20000):fill(0),
        labelFine = torch.IntTensor(20000):fill(0),
        indicator = torch.IntTensor(20000,2):fill(1),
        size = function () return 20000 end
    } --20000
    train_coarse_tmp.indicator[{{}, {1}}]:fill(0)
    train_coarse_tmp.indicator[{{}, {2}}]:fill(1)

    -- Initialize val dataset
    val_tmp = {
        data = torch.DoubleTensor(10000,3,32,32):fill(0),
        labelCoarse = torch.IntTensor(10000):fill(0),
        labelFine = torch.IntTensor(10000):fill(0),
        indicator = torch.IntTensor(10000,2):fill(1),
        size = function () return 10000 end
    } --10000

    -- Assign data into the subsets
    fine_perm = torch.randperm(500)
    val_count = 0
    train_fine_count = 0
    train_coarse_count = 0
    for i=1,100 do
       count = 0
       for j=1,50000 do
           if cifar100.trainData.labelFine[j] == i then
              count = count + 1
              if fine_perm[count] <= 50 or fine_perm[count] > 450 then -- validation set
                 val_count = val_count + 1
                 val_tmp.data[val_count] = cifar100.trainData.data[j]
                 val_tmp.labelFine[val_count] = cifar100.trainData.labelFine[j]
                 val_tmp.labelCoarse[val_count] = cifar100.trainData.labelCoarse[j]
              elseif fine_perm[count] > 50 and fine_perm[count] <= 250 then -- coarse set
                 train_coarse_count = train_coarse_count + 1
                 train_coarse_tmp.data[train_coarse_count] = cifar100.trainData.data[j]
                 train_coarse_tmp.labelFine[train_coarse_count] = cifar100.trainData.labelFine[j]
                 train_coarse_tmp.labelCoarse[train_coarse_count] = cifar100.trainData.labelCoarse[j]
              else -- fine set
                 train_fine_count = train_fine_count + 1
                 train_fine_tmp.data[train_fine_count] = cifar100.trainData.data[j]
                 train_fine_tmp.labelFine[train_fine_count] = cifar100.trainData.labelFine[j]
                 train_fine_tmp.labelCoarse[train_fine_count] = cifar100.trainData.labelCoarse[j]   
              end
           end         
       end        
    end

    torch.save(coarse_path, train_coarse_tmp)
    torch.save(fine_path, train_fine_tmp)
    torch.save(val_path, val_tmp)
end

trainDataCoarse = torch.load(coarse_path)
trainDataFine = torch.load(fine_path)
testData = torch.load(val_path)


