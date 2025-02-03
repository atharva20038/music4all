import torch
import torch.nn as nn
import math

class BottleNeckAdapter(nn.Module):
  def __init__(self, in_channels, adapter_dim, num_heads=4):

    """
    Args:
        in_channels (int): Number of input channels.
        adapter_dim (int): Dimensionality of the adapter.
        num_heads (int): Number of attention heads in the self-attention layer.
    """
    super().__init__()
    print(f"Using BottleNeckAdapter with {adapter_dim} dimensions.")
    print(f"In channels: {in_channels}")
    self.adapter_proj_down = nn.Linear(in_channels, adapter_dim)
    self.adapter_proj_up = nn.Linear(adapter_dim, in_channels)
    self.attn = nn.MultiheadAttention(adapter_dim, num_heads)
    self.act = nn.ReLU()  # Choose your desired activation function (e.g., GELU, SiLU)
    # self.norm = nn.GroupNorm(1, adapter_dim)  # Apply GroupNorm for regularization

  def forward(self, x):
    # Down-projection
    print(f"Input shape: {x.shape}")
    x = self.adapter_proj_down(x)
    x = self.act(x)
    # x = self.norm(x)
    # Self-attention
    x = x.permute(1, 0, 2)  # MultiheadAttention expects (L, N, E) instead of (N, L, E)
    x, _ = self.attn(x, x, x)
    x = x.permute(1, 0, 2)
    # Up-projection
    x = self.adapter_proj_up(x)
    x = self.act(x)
    # x = self.norm(x)
    return x

class LoRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=16, dropout=0.0):
        """
        Args:
            in_features (int): Input dimensionality (size of the embedding).
            out_features (int): Output dimensionality (size of the output embedding).
            r (int): The rank of the low-rank decomposition (controls adapter size).
            alpha (int): Scaling factor applied to the low-rank adaptation.
            dropout (float): Dropout probability applied to the output of the adapter.
        """
        super(LoRAAdapter, self).__init__()
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA down-projection (reduce dimension to low-rank representation)
        self.down_proj = nn.Linear(in_features, r, bias=False)
        
        # LoRA up-projection (project back to original dimension)
        self.up_proj = nn.Linear(r, out_features, bias=False)
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize down and up projection layers
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        # Down-project input to lower rank
        down = self.down_proj(x)  # Shape: (batch_size, seq_len, r)
        
        # Up-project back to original dimension
        up = self.up_proj(down)  # Shape: (batch_size, seq_len, out_features)
        
        # Apply scaling and dropout
        output = self.dropout(up) * self.scaling
        
        return output

class Adapter(nn.Module):
    def __init__(self, in_features, out_features, adapter_dim=128, num_heads=4, dropout=0.0, adapter_type="bottleneck"):
        """
        Args:
            in_features (int): Input dimensionality (size of the embedding).
            out_features (int): Output dimensionality (size of the output embedding).
            adapter_dim (int): Dimensionality of the adapter.
            num_heads (int): Number of attention heads in the self-attention layer.
        """
        super().__init__()
        if adapter_type == "bottleneck":
            self.adapter = BottleNeckAdapter(in_features, adapter_dim, num_heads)

        elif adapter_type == "lora":
            self.adapter = LoRAAdapter(in_features, out_features, r=4, alpha=16, dropout=dropout)

        print(f"Using {adapter_type} adapter with {adapter_dim} dimensions.")

    def forward(self, x):
        return self.adapter(x)

# # Define AdaGroupNorm (assuming it's defined elsewhere)
# class AdaGroupNorm(nn.Module):
#     def __init__(self, num_channels, num_groups, num_groups_per_channel=1):
#         super(AdaGroupNorm, self).__init__()
#         self.gn = nn.GroupNorm(num_groups, num_channels)
        
#     def forward(self, x):
#         return self.gn(x)

# Test function to evaluate the Adapter with a 2D input
def test_adapter_module():
    batch_size = 8
    seq_length = 16
    in_features = 64
    out_features = 64
    adapter_dim = 32
    num_heads = 4
    dropout = 0.1

    # Create a random input tensor (batch_size, seq_length, in_features)
    input_tensor = torch.randn(batch_size, seq_length, in_features)

    # Test with BottleNeckAdapter
    print("Testing BottleNeckAdapter...")
    bottleneck_adapter = Adapter(in_features, out_features, adapter_dim, num_heads, dropout, adapter_type="bottleneck")
    output_bottleneck = bottleneck_adapter(input_tensor)
    print(f"BottleNeckAdapter output shape: {output_bottleneck.shape}\n")

    # Test with LoRAAdapter
    print("Testing LoRAAdapter...")
    lora_adapter = Adapter(in_features, out_features, adapter_dim, num_heads, dropout, adapter_type="lora")
    output_lora = lora_adapter(input_tensor)
    print(f"LoRAAdapter output shape: {output_lora.shape}\n")

if __name__ == "__main__":
    test_adapter_module()
