import torch


def simplex_projection(v, s=1):
    """
    Projects each row of v onto the simplex of specified radius s.

    Reference: Duchi et al., "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008

    Args:
        v (torch.Tensor): Input tensor of shape (..., n)
        s (float): Radius of the simplex (default 1)
    Returns:
        torch.Tensor: Projected tensor of the same shape as v
    """
    # Flatten all but last dimension for batch processing
    orig_shape = v.shape
    v = v.reshape(-1, v.shape[-1])
    # Sort v in descending order
    u, _ = torch.sort(v, dim=1, descending=True)
    # Compute cumulative sum
    cssv = torch.cumsum(u, dim=1) - s
    # Create an index tensor 1..n
    ind = torch.arange(1, v.shape[1] + 1, device=v.device).float()
    # Find the rho value for each row
    cond = u > (cssv / ind)
    rho = cond.sum(dim=1) - 1
    # Gather theta for each row
    theta = cssv[torch.arange(v.shape[0]), rho] / (rho + 1).float()
    # Compute projection
    w = torch.clamp(v - theta.unsqueeze(-1), min=0)
    # Restore original shape
    return w.reshape(orig_shape)


if __name__ == "__main__":
    v = torch.tensor([0.25, 0, 0.5, 0.4], dtype=torch.float32)
    projected_v = simplex_projection(v)
    print(projected_v)
    print("Sum:", projected_v.sum())
