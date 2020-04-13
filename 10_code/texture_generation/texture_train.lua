require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'optim'

display = require('display')

require 'src/utils'
require 'src/descriptor_net'

local cmd = torch.CmdLine()

cmd:option('-texture_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'Layers to attach texture loss.')
cmd:option('-texture', '/hdd/2019-bass-connections-aatb/texture_v1/texture_nets/test/chicago9_roof.tif', 'Style target image')

cmd:option('-learning_rate', 1e-1)
cmd:option('-num_iterations', 1500)

cmd:option('-batch_size', 16)

cmd:option('-image_size', 256) -- how do we preserve aspect ratio of input images  
cmd:option('-noise_depth', 3, 'Number of channels of the input Tensor.')

cmd:option('-gpu', 0, 'Zero indexed gpu number.')
cmd:option('-tmp_path', 'data/irg/', 'Directory to store intermediate results.')
cmd:option('-model_name', '', 'Path to generator model description file.')

cmd:option('-normalize_gradients', 'false', 'L1 gradient normalization inside descriptor net. ')
cmd:option('-vgg_no_pad', 'false')

cmd:option('-proto_file', 'data/pretrained/VGG_ILSVRC_19_layers_deploy.prototxt', 'Pretrained')
cmd:option('-model_file', 'data/pretrained/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn')

cmd:option('-circular_padding', 'true', 'Whether to use circular padding for convolutions. Use by default.')

params = cmd:parse(arg)

params.normalize_gradients = params.normalize_gradients ~= 'false'
params.vgg_no_pad = params.vgg_no_pad ~= 'false'
params.circular_padding = params.circular_padding ~= 'false'
params.texture_weight = 1

if params.backend == 'cudnn' then
  require 'cudnn'
  cudnn.fastest = true
  cudnn.benchmark = true
  backend = cudnn
else
  backend = nn
end

-- Whether to use circular padding
if params.circular_padding then
  conv = convc
end

cutorch.setDevice(params.gpu+1)

net_input_depth = params.noise_depth
num_noise_channels = params.noise_depth

-- Define model
local net = require('models/' .. params.model_name):cuda()
local descriptor_net, _, texture_losses = create_descriptor_net()

----------------------------------------------------------
-- feval
----------------------------------------------------------

iteration = 0

-- dummy storage, this will not be changed during training
inputs_batch = torch.Tensor(params.batch_size, net_input_depth, params.image_size, params.image_size):uniform():cuda()

local parameters, gradParameters = net:getParameters()
loss_history = {}
function feval(x)
  iteration = iteration + 1
  
  if x ~= parameters then
      parameters:copy(x)
  end
  gradParameters:zero()
  
  -- forward
  local out = net:forward(inputs_batch)
  descriptor_net:forward(out)
  
  -- backward
  local grad = descriptor_net:backward(out, nil)
  net:backward(inputs_batch, grad)
  
  -- collect loss
  local loss = 0
  for _, mod in ipairs(texture_losses) do
    loss = loss + mod.loss
  end
  
  table.insert(loss_history, {iteration,loss/params.batch_size})
  print(iteration, loss/params.batch_size)

  return loss, gradParameters
end
----------------------------------------------------------
-- Optimize
----------------------------------------------------------
print('        Optimize        ')

optim_method = optim.adam
state = {
   learningRate = params.learning_rate,
}


for it = 1, params.num_iterations do
  
  -- Optimization step
  optim_method(feval, parameters, state)

  -- Visualize
  if it%10 == 0 then
    collectgarbage()

    local output = net.output:clone():double()

    local imgs  = {}
    for i = 1, output:size(1) do
      local img = deprocess(output[i])
      table.insert(imgs, torch.clamp(img,0,1))
      image.save(params.tmp_path .. 'train' .. i .. '_' .. it .. '.png',img)
    end

    display.image(imgs, {win=params.gpu, width=params.image_size*3, title = params.gpu})
    display.plot(loss_history, {win=params.gpu+4, labels={'iteration', 'Loss'}, title='Gpu ' .. params.gpu .. ' Loss'})
  end
  
  if it%300 == 0 then 
    state.learningRate = state.learningRate*0.8 
  end

  -- Dump net, the file is huge
  if it%200 == 0 then 
    torch.save(params.tmp_path .. 'model' .. it .. '.t7', net:clearState())
  end
end
-- Clean net and dump it, ~ 500 kB
torch.save(params.tmp_path .. 'model.t7', net:clearState())


