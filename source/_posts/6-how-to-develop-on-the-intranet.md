---
title: 技术相关｜在内网搭建开发环境
date: 2022-09-09 23:12:41
cover: false
categories:
 - Techniques
tags:
 - Linux
---

最近依然是在绝赞跑代码做实验，然而因为学长需要在我用的服务器上 debug，所以我换到了组里的另一台机器上。但是这台机器有一个非常坑爹的地方，就是它没法连接外网。我用 `ssh` 连了一下发现并不能连通，`ping` 也全是超时，最后先 `ssh` 到了组里的另一台服务器上，又从这台服务器在内网连，才成功登录。

登录上去后测试了一下网络环境，发现果然只能连接内网，不能连外网。试图定位了一下原因但是无果（感觉很有可能是硬件问题，比如网卡坏了或者机房网线掉了），于是我最后决定通过搭建一系列代理来实现内外网相互连通。下面来记录一下配置的过程，出于安全相关的考虑，以下使用的 ip 地址均为编造的示例地址。

# 配置正向代理

在内网上，我已经有了一台可以连接外网的机器 A（ip 地址为 `192.168.10.1`），以及一台无法连接外网，但可以通过内网进行访问的机器 B（ip 地址为 `192.168.10.2`）。

为了使机器 B 能够访问外网，一个比较直接的方法是以机器 A 作为代理，将流量包在外网和机器 B 之间进行转发。我使用的是轻量级 HTTP/HTTPS 代理 `tinyproxy`。

首先安装 `tinyproxy`，可以使用 `apt-get` 一键获取：

```shell
sudo apt update
sudo apt install tinyproxy
```

然后对 `tinyproxy` 进行配置：

```shell
sudo vim /etc/tinyproxy/tinyproxy.conf
```

主要需要配置的只有两个地方，第一个是端口号，第二个是出站规则。

首先找到名为 `Port` 的配置，将其改为一个没有被占用的端口号，例如 `12345`；然后找到配置文件中类似 `Allow 127.0.0.1` 的一行，将这行注释掉，表示可以接受所有 ip 的访问。为了简便起见，其他的配置文件我都没有做改动。

配置结束之后即可启动服务，以下是用于启动/重启/停止服务的命令：

```shell
sudo systemctl start tinyproxy.service
sudo systemctl restart tinyproxy.service
sudo systemctl stop tinyproxy.service
```

启动后用 `sudo systemctl status tinyproxy.service` 查看服务状态。到这一步，代理服务器就已经搭建完毕，只需要再配置客户端即可。

连接机器 B，在所使用 shell 的 rc 文件（例如 `~/.bashrc`、`~/.zshrc` 等）中加入以下命令并重新登录。注意 ip 地址与端口号应当根据实际情况设定。

```shell
export http_proxy=http://192.168.10.1:12345
export https_proxy=http://192.168.10.1:12345
```

值得一提的是，在 pc 上使用 shadowsocks、clash 等代理客户端时，也可以用这种方式控制命令行使用的代理地址。（不过这种情况下，会多一条类似 `export all_proxy=socks5://192.168.10.1:12345` 的命令，在服务器上不用加这一条是因为 `tinyproxy` 只支持 HTTP/HTTPS 代理。）

至此代理服务器就配置完毕了，可以使用 `wget www.baidu.com` （由于 `ping` 使用的是 ICMP 协议，所以在这种场景下 `ping` 依然无法联通 ）测试网络连通性，如果能够正确下载 `index.html` 文件，就表示网络已经连通。如果无法连通，应当检查代理服务器是否能联网，以及上述步骤是否全部配置正确。

{% note warning flat %}

在校园网环境下，能成功下载 `index.html` 并不意味着成功联通，因为连接可能被校园网网关重定向到登陆认证界面。因此在下载结束后还需要 `cat index.html` 来观察是否下载的是百度的页面，如果发生重定向，需要先将代理服务器连接外网。

{% endnote %}

除了最基础的配置之外，还有一些常用的配置可供参考，例如可以在 `rc` 文件中加入如下的 alias，让 `curl` 命令默认使用代理：

```shell
alias curl="curl -x http://192.168.10.1:12345"
```

以及在 `~/.ssh/config` 中加入如下配置，使 GitHub/Gitee 可以通过 ssh 联通：

```
Host Jump
  HostName 192.168.10.1
  User $username

Host github.com
  User git
  ProxyCommand ssh -W %h:%p Jump

Host giee.com
  User git
  ProxyCommand ssh -W %h:%p Jump
```

{% note warning flat %}

作为一种折中的方法，这种方法也存在一些缺陷。因为只对 HTTP/HTTPS 进行了代理，使用其他传输协议的功能（例如 `ping` 的某些功能）可能无法正常运行。但对于日常开发的需要来说，这就已经足够使用了。

{% endnote %}

# 配置跳板机

虽然现在内网的机器可以正常联网了，但登录还是很不方便，需要先登录机器 A，再从 A 登录 B 机器，这样比较麻烦。因此，还需要对跳板机进行配置。

在本地的 `~/.ssh/` 目录下找到 `config` 文件（如果没有就创建一个），然后写入以下内容：

```
Host Jump
  HostName 192.168.10.1
  User $username

Host Target
  HostName 192.168.0.2
  User $username
  ProxyCommand ssh -W %h:%p Jump
```

注意上述的 `User` 字段应该填写服务器上使用的用户名。然后，再将自己本地的 rsa public key 上传到机器 B，并写入 `~/.ssh/authorized_keys`。这样，就可以直接在本地使用类似 `ssh Target` 的命令直接登录到目标机器上。

> 折腾了半个晚上终于弄好了，好在这台机器的训练速度很快，给了我一点慰藉（）