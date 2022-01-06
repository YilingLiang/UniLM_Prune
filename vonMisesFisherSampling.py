import torch
import numpy as np
import torch.nn as nn

class vonMisesFisherSampling(nn.Module):
    """von Mises Fisher分布重参数
    通过累积概率函数的逆和预计算来实现最简单vMF分布采样
    链接：https://kexue.fm/archives/8404
    改写自苏的keras实现
    """

    def __init__(self, kappa=0.4, num_caches=10 ** 7, dims=768):
        super(vonMisesFisherSampling, self).__init__()
        self.kappa = kappa  # 超参数k
        self.num_caches = num_caches  # 预先存多少
        self.dims = dims

        self.register_buffer('pw_samples', torch.zeros(self.num_caches, 1))
        self.pw_samples = torch.from_numpy(self.init())
        self.pw_samples = self.pw_samples.reshape(num_caches, 1) # 这里要 reshape 一下

    def init(self):
        """
        事先把w的采样存下来，10**7个
        """
        x = np.linspace(-1, 1, self.num_caches + 2)[1:-1]
        y = self.kappa * x + np.log(1 - x ** 2) * (self.dims - 3) / 2
        y = np.cumsum(np.exp(y - y.max()))
        return np.interp((x + 1) / 2, y / y[-1], x)

    def forward(self, mu):
        shape = mu.size()
        print("mu shape: ", shape)
        print("pw_sample shape: ", self.pw_samples.shape)
        # 采样w
        idxs = torch.empty(shape[0], 1)
        print("idx shape: ", idxs.shape)
        nn.init.uniform_(idxs, 0, self.num_caches)
        # idxs = idxs.int()
        idxs = idxs.long()
        idxs.to(device=mu.device, dtype=torch.int64)
        print(idxs)
        w = torch.gather(self.pw_samples, dim=0, index=idxs)  # pw_samples (num_caches, 1), idxs (bsz, 1) -> w (bsz, 1)
        # 采样z
        eps = torch.normal(0, 1, size=shape).to(mu.device)  # (bsz, dim)
        nu = eps - torch.sum(eps * mu, dim=1, keepdim=True) * mu
        nu = nn.functional.normalize(nu, dim=1)
        return w * mu + (1 - w ** 2) ** 0.5 * nu

    # def build(self, input_shape):
    # 所以一开始这个就是w的采样就是存好的
    # 这里的input_shape是keras一开始建图的时候提供的，形状为(dim, ),即没有batch_size, 只有tensor的维度
    #     super(vonMisesFisherSampling, self).build(input_shape)
    #     self.pw_samples = self.add_weight(
    #         shape=(self.num_caches,),
    #         initializer=self.initializer(input_shape[-1]),
    #         trainable=False,
    #         name='pw_samples'
    #     )

    # def initializer(self, dims):
    #     def init(shape, dtype=None):
    #         x = np.linspace(-1, 1, shape[0] + 2)[1:-1]
    #         y = self.kappa * x + np.log(1 - x ** 2) * (dims - 3) / 2
    #         y = np.cumsum(np.exp(y - y.max()))
    #         return np.interp((x + 1) / 2, y / y[-1], x)
    #
    #     return init
    #
    # def call(self, inputs):
    #     mu = inputs
    #     # 采样w
    #     idxs = K.random_uniform(
    #         K.shape(mu[..., :1]), 0, self.num_caches, dtype='int32'
    #     )
    #     w = K.gather(self.pw_samples, idxs)
    #     # 采样z
    #     eps = K.random_normal(K.shape(mu))
    #     nu = eps - K.sum(eps * mu, axis=1, keepdims=True) * mu
    #     nu = K.l2_normalize(nu, axis=-1)
    #     return w * mu + (1 - w ** 2) ** 0.5 * nu
if __name__ == "__main__":
    vmf = vonMisesFisherSampling(kappa=0.4, num_caches=10 ** 7, dims=32)
    mu = torch.randn((4, 32))
    vmfsamples = vmf(mu)
    print("vmf: ", vmfsamples)
    print(vmfsamples.shape)

