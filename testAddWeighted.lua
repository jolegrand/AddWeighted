dofile("addWeighted.lua")

local size = 10
local maxsum = 10

aw = nn.AddWeighted(size, maxsum)

local net = nn.Sequential()
net:add(aw)
net:add(nn.Linear(size,1))

local criterion = nn.MSECriterion()

local epsilon = 0.0001

local input = torch.rand(math.random(maxsum), size)
print(input)


for i=1,input:size(1) do
   for j=1,input:size(2) do
      local backup = input:clone()
      local score1 = net:forward(input)[1][1]
      
      input[i][j] = input[i][j] + epsilon
      local score2 = net:forward(input)[1][1]

      --print("score 1 : " .. score1)
      --print("score 2 : " .. score2)
      local deriv = (score2 - score1) / epsilon
      --print("deriv : " .. deriv)
      
      local grad = torch.Tensor({{1}})
      net:backward(input, grad)
      local g = aw.gradInput[i][j]
      --print("print grad : " .. g)
      print("error : " .. math.abs(deriv-g))
      if math.abs(deriv-g)>0.00001 then error("error in grad") end
   end
end