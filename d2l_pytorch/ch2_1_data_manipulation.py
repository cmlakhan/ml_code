import torch


x = torch.arange(12)
x
## dimensions of X
x.shape

## number of elements in x
x.numel()

## change the shape of the vector -1 can be used to automatically infer dimension
x = x.reshape(3, 4)
x
### or
x = x.reshape(-1, 4)
### or
x = x.reshape(3, -1)

## Torch tensor with zeros
torch.zeros(2, 3, 4)

## Torch tensor with ones
torch.ones((2, 3, 4))

## Sample from a standard gaussian
torch.randn(3, 4)

## Specify the exact values
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

## Elements wise operations
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

### element-wise addition
x + y
### element-wise subtraction
x - y
### element-wise multiplication
x * y
### element-wise division
x / y
### element-wise exponentiation
x ** y
### element-wise unary operation
torch.exp(x)

## Concatenate two vectors
x = torch.arange(12, dtype=torch.float32).reshape((3,4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
### Dimension 0
torch.cat((x, y), dim=0)
### Dimension 1
torch.cat((x, y), dim=1)

## Equality operator
x
y
x == y


## Sum