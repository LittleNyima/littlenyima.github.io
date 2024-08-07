---
title: 技术相关｜Linux 虚拟机的安全加固
date: 2023-07-12 01:01:42
cover: false
categories:
 - Techniques
tags:
 - Linux
 - Security
---

近期正在部署一个服务，刚刚准备开放公网端口，在开放之前先简单做一下安全加固，以我使用的 Ubuntu 22.04.1 LTS 系统为例。

# ssh 加固

## 更换登录端口

默认的 ssh 端口为 22，为了防止端口扫描&暴力破解，我将 ssh 端口修改为 27001（以此为例）。首先编辑 `/etc/ssh/sshd_config`：

```
# Port 22
Port 27001
```

保存后用 `service sshd restart` 重启 `sshd` 服务，即可重置 ssh 端口。

## 禁用 root 登陆

随后需要禁止 `root` 账户登陆，首先创建普通用户：`useradd -d /home/littlenyima -m littlenyima`，然后使用 `passwd littlenyima` 为新用户创建一个密码。把用户添加到 sudo 用户组：`visudo /etc/sudoers`

```
root	ALL=(ALL:ALL) ALL
littlenyima	ALL=(ALL:ALL) ALL
```

然后编辑 `/etc/ssh/sshd_config`，禁用 `root` 账户登陆，保存后重启 `sshd` 服务：

```
# PermitRootLogin yes
PermitRootLogin no
```

## 禁用密码登录

首先确保可以使用密钥登录，先在本地使用 `ssh-keygen` 生成 rsa 公钥与密钥，然后将本地的公钥（默认位置位于 `~/.ssh/id_rsa.pub`）拷贝到虚拟机的 `~/.ssh/authorized_keys`，即可使用密钥登录。

然后修改 `/etc/ssh/sshd_config`，禁用密码登录，保存后重启 `sshd` 服务：

```
# PasswordAuthentication yes
PasswordAuthentication no
```

为了方便本地登录，可以在 `~/.ssh/config` 进行配置：

```
Host RemoteVM
  HostName <ip address or domain name>
  User littlenyima
  Port 27001
```

> 更新：经过测试发现经过这一设置后，仍然可以用密码登录服务器，具体原因有待排查

经过上述一系列操作，ssh 被攻破的概率基本上是很小的了，除非攻击者直接拿到了我的本地设备（都线下了，直接真人快打可能效率更高一点x），否则基本上无法登录到我的虚拟机上。

后续的其他加固待我弄好再继续更新（）