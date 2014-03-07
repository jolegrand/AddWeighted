require("nn")

local AddWeighted, parent = torch.class('nn.AddWeighted', 'nn.Module')

function AddWeighted:__init(insize, maxSize)
   parent.__init(self)
   self.insize = insize
   self.maxSize = maxSize
   self.weight = torch.Tensor(1,maxSize):fill(1/maxSize)
   self.output:resize(1,insize)
   self.gradInput:resize(maxSize,insize)
   self:reset()
end

function AddWeighted:reset()
   self.weight:resize(1,self.maxSize):fill(1)
   self.weight:div(self.maxSize)
end

function AddWeighted:updateOutput(input)
   if self.weight:size(2)~=input:size(1) then 
      self.weight:resize(1,input:size(1)):fill(1/input:size(1))
   end
   self.output:mm(self.weight, input)
   return self.output
end

function AddWeighted:updateGradInput(input, gradOutput)
   self.gradInput:resize(self.weight:size(2), self.insize)
   self.gradInput:mm(self.weight:t(), gradOutput)
   return self.gradInput
end

function AddWeighted:accGradParameters(input, gradOutput, scale)
   --
end