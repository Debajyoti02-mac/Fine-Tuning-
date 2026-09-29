import torch 
from torch import nn

# Data  
x = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[2.0],[4.0],[6.0],[8.0]])

print(f'the first tensor is : \n {x}')
print(f'the second tensor : \n {y}')

# Model 
model = nn.Linear(1,1)

# Loss 
loss_fun = nn.MSELoss()

# optimizer 
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.01)

# training loop 
for epochs in range(5000):
    
    # Prediction 
    prediction = model(x)
    
    # loss 
    loss = loss_fun(y,prediction)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
print(model(torch.tensor(([5.0]))))
    