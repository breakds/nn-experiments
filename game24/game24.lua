require 'nn'

function check24(input)
   if #input == 1 then
      if torch.round(input[1]) == 24 then
         return true
      else
         return false
      end
   else
      for i = 1, #input - 1 do
         for j = i + 1, #input do
            new_input = {}
            for k = 1, #input do
               if (k ~= i) and (k ~= j) then
                  new_input[#new_input + 1] =  input[k]
               end
            end
            new_input[#new_input + 1] = input[i] + input[j]
            if check24(new_input) then return true end
            new_input[#new_input] = input[i] * input[j]
            if check24(new_input) then return true end
            new_input[#new_input] = input[i] - input[j]
            if check24(new_input) then return true end
            new_input[#new_input] = input[j] - input[i]
            if check24(new_input) then return true end
            if input[j] ~= 0 then
               new_input[#new_input] = input[i] / input[j]
               if check24(new_input) then return true end
            end
            if input[i] ~= 0 then
               new_input[#new_input] = input[j] / input[i]
               if check24(new_input) then return true end
            end
         end
      end
      return false
   end
end


function ToVec(input)
   local vec = torch.zeros(10)
   for i = 1, #input do
      vec[input[i] + 1] = vec[input[i] + 1] + 1
   end
   return vec
end

-- Prepare dataset

dataset = {}

for i = 0, 9 do
   for j = 0, 9 do
      for k = 0, 9 do
         for l = 0, 9 do
            local v = {i, j, k, l}
            local c = -1
            if check24(v) then c = 1 end
            dataset[#dataset + 1] = {ToVec(v), torch.Tensor({c})}
         end
      end
   end
end

print("dataset prepared.")

-- Prepare training and validation set

torch.manualSeed(773)

training_set = {}

function training_set:size()
   return 4000
end

validate_set = {}

function validate_set:size()
   return #dataset - training_set:size()
end

shuffle = torch.randperm(#dataset)

for i = 1, #dataset do
   if i <= training_set:size() then
      training_set[i] = dataset[shuffle[i]]
   else
      validate_set[i - training_set:size()] = dataset[shuffle[i]]
   end
end

print("training/validate set prepared.")

-- Define Neural Network

model = nn.Sequential()
model:add(nn.Linear(10, 20))
model:add(nn.Tanh())
model:add(nn.Linear(20, 10))
model:add(nn.Tanh())
model:add(nn.Linear(10, 1))
print(model)

-- Training
loss = nn.MSECriterion()
training = nn.StochasticGradient(model, loss)
training.learningRate = 0.01
training.maxIteration = 100
training:train(training_set)

do 
   local count = 0
   for i = 1, #validate_set do
      local pass = false
      if (model:forward(validate_set[i][1]) * validate_set[i][2]) > 0 then
         count = count + 1
         pass = true
      end
      if pass then
         print("testing " .. i .. "/" .. #validate_set .. " pass")
      else
         print("testing " .. i .. "/" .. #validate_set)
      end
   end
   print("final: " .. count .. "/" .. #validate_set)
end




