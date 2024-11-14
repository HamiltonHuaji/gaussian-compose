# gaussian-compose

## gaussian_compose.ops

```python
def indexed_transform(trans: torch.Tensor, rotors: torch.Tensor, means: torch.Tensor, quats: torch.Tensor, indices: torch.Tensor, implementation="slang") -> Tuple[torch.Tensor, torch.Tensor]:
    """Fetch the global transformation parameters trans & rotors by indices, and apply to local transformation parameters means & quats

    pass
```

全局变换:
+ trans: (n, 3)
+ rotors: (n, 4)

局部变换:
+ means: (m, 3)
+ quats: (n, 3)


