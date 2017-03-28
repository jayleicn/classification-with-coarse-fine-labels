local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 cifar100 Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-cache', './results/', 'subdirectory in which to save/log experiments')
    cmd:option('-data', './data/', 'Home of cifar100 dataset')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-backend',     'cunn', 'Options: cunn | cudnn | nn')
    ------------- Data options ------------------------
    cmd:option('-dataSize',        20000, 'number of Fine/Coarse training images')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         55,    'Number of total epochs to run')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       64,   'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'nin_coarse_fine', 'Options: nin_coarse_fine, VGG_C100_coarse_fine')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache,
                            cmd:string(opt.netType, opt,
                                       {netType=true, retrain=true, optimState=true, cache=true, data=true}))
    -- add date/time
    cur_time= os.date():gsub(' ','')  -- will return 2 values, only the first is required here.
    opt.save = paths.concat(opt.save, '' .. cur_time)
    return opt
end

return M
