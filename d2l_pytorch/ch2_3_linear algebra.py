import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])

x + y, x * y, x / y, x**y


x = torch.arange(4)
x


x[3]

len(x)

x.shape


A = torch.arange(20).reshape(5, 4)
A

A.T

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B

B == B.T


## Create a Tensor
X = torch.arange(24).reshape(2, 3, 4)
X


A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B

## Element wise addition is called hadamard product
A * B

## Adding a scale to a tensor
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape


## Reduction
x = torch.arange(4, dtype=torch.float32)
x, x.sum()


## Sum all elements of a matrix
A.shape, A.sum()


## You can specify the axis where you want to sum up
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape

A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape

## Reducing along two axis
A.sum(axis=[0, 1])  # Same as `A.sum()`

## Mean, sum, and number of elements
A.mean(), A.sum() / A.numel()

## Keep the dimensions of the sum so that it still has the original shape
sum_A = A.sum(axis=1, keepdims=True)
sum_A

## then you can broadcast
A / sum_A


## Cumulative sum across the 0 axis (rows)
A.cumsum(axis=0)


## Dot product
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)

or
torch.sum(x * y)


## Matrix Vector Products
A.shape, x.shape, torch.mv(A, x)

## Matrix Matrix Multiplication
B = torch.ones(4, 3)
torch.mm(A, B)


## Norm of a vector
u = torch.tensor([3.0, -4.0])
torch.norm(u)

## L1 norm
torch.abs(u).sum()

## Frobenius Norm (l2 norm of matrix)
torch.norm(torch.ones((4, 9)))
