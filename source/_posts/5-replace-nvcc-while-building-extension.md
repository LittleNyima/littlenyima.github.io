---
title: 技术相关 | 编译 Pytorch 扩展时替换 nvcc 版本
date: 2022-08-27 23:16:29
cover: false
categories:
 - Techniques
tags:
 - Pytorch
---

今天解决了一个有点复杂的环境问题，记录一下解决的过程。

在复现 [ReferFormer](https://github.com/wjn922/ReferFormer) 时，需要编译一个 Deformable Attention 算子。在编译的过程中，`nvcc` 使用了一个叫做 `--generate-dependencies-with-compile` 的 flag。非常不幸的是，我现有开发环境中的 `nvcc` 并不支持这一个 flag，导致我无法编译这个算子。为了解决这个问题，我首先确定了现有的 `nvcc` 版本：

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Fri_Feb__8_19:08:17_PST_2019
Cuda compilation tools, release 10.1, V10.1.105
```

经过查找资料，我发现这个 flag 是在 `nvcc-10.2` 版本中被引入的。因此我决定替换运行环境中的 `nvcc`。

由于无法获得 sudo 权限，我只能在用户权限之内安装新版本的 `nvcc`。所幸，即使在没有管理员权限的情况下，新版本的 `nvcc` 也可以使用 `conda` 很方便地安装：

```shell
conda install -c nvidia cuda-nvcc
```

这个命令会将 `nvcc` 安装到 `$CONDA_DEFAULT_ENV/bin/nvcc` 的位置，只需要替换编译时使用的 `nvcc` 就大功告成了。然而，当我在 `~/.bashrc` 中加入：

```shell
alias nvcc='$CONDA_DEFAULT_ENV/bin/nvcc'
```

再运行 `which nvcc` 和 `nvcc -V` 时，我发现 `nvcc` 并没有被替换，再编译发现使用的也还是原有的 `nvcc`。我猜测在 Pytorch 实现时，可能是先找到了 `CUDA_HOME`，再在这个目录下找到了 `nvcc`。

观察 `build` 目录，我发现目录里生成了一个 `build.ninja`，所以不难猜测，编译 c++ 拓展的时候，内部的逻辑是：Pytorch 生成 `build.ninja` 文件、setuptools 根据这个文件调用 `ninja`，来进行编译。因此要替换 `nvcc`，只需要找到这个 build 脚本生成的位置即可。

找到环境里的 `torch.utils.cpp_extension`，在文件里搜索 `nvcc` 和 `ninja`，发现可以找到一个叫做 `_write_ninja_file` 的函数，将这个函数中 `config.append(f'nvcc = {nvcc}')` 的 `{nvcc}` 替换为需要使用的 `nvcc` 路径，即可解决问题。

> p.s. 不得不说 cuda 版本什么的永远都是配环境的时候最痛的部分…