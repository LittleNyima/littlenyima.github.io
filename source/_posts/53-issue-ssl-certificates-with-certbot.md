---
title: 技术相关｜使用 Certbot 为通配符域名签发 SSL 证书
date: 2025-05-26 17:57:41
cover: false
categories:
 - Techniques
tags:
 - Linux
 - SSL
 - Certbot
---

近期在腾讯云给 OSS 配置了自定义域名，并且因为我有两个桶，所以配置了两个不同的子域名。又为了让文件能支持 HTTPS 访问，需要在腾讯云后台给域名配置 SSL 证书。但非常坑的是腾讯云的 SSL 证书不仅需要花钱买而且非常贵，因此为了节能减排最后我选择在 Let's Encrypt 自己签发一个证书。

Let's Encrypt 推荐的客户端是 Certbot，而为了让多个子域名都能使用同一个证书，需要为通配符域名（也就是一个类似 `*.my-domain.com` 形式的域名）签发证书，这需要使用 DNS-01 模式进行签发。

从具体操作来说，首先需要访问 [Certbot 的官方网站](https://certbot.eff.org/)并且选择正确的选项。对于我来说，我并不需要在本地部署证书，因此我选的是 My HTTP website is running **Other** on **Linux (snap)**。选择之后一步步跟着官方的教程安装 `certbot`，对于我来说命令是下边这些，其他系统的命令可能有所不同：

```shell
sudo apt-get remove certbot
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot
```

安装成功后运行下边的命令即可开始签发过程：

```shell
sudo certbot certonly --manual --preferred-challenges dns -d '*.my-domain.com'
```

如果前边的步骤都没有问题的话，运行之后会看到一个类似下面形式的输出：

```
Saving debug log to /var/log/letsencrypt/letsencrypt.log
Requesting a certificate for *.my-domain.com

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Please deploy a DNS TXT record under the name:

_acme-challenge.my-domain.com.

with the following value:

4uSjh1kbHQFrozxG1F9bK2UF6UxNm893qYE8n7pE6dQ

Before continuing, verify the TXT record has been deployed. Depending on the DNS
provider, this may take some time, from a few seconds to multiple minutes. You can
check if it has finished deploying with aid of online tools, such as the Google
Admin Toolbox: https://toolbox.googleapps.com/apps/dig/#TXT/_acme-challenge.my-domain.com.
Look for one or more bolded line(s) below the line ';ANSWER'. It should show the
value(s) you've just added.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Press Enter to Continue
```

根据上面的指示，需要手动添加一条名字为 `_acme-challenge` 的 DNS 记录，类型为 `TXT`，内容则是上述信息给出的那串超长随机字符串。添加之后去访问 Google 管理员工具箱的地址，看到这条记录出现之后就可以回车开始进行下一个步骤。需要注意的是，这个操作需要有域名对应的 DNS 解析权限，如果没有的话是无法添加这条记录的。

如果一切操作都正确的话，会看到下面的输出：

```
Successfully received certificate.
Certificate is saved at: /etc/letsencrypt/live/my-domain.com/fullchain.pem
Key is saved at:         /etc/letsencrypt/live/my-domain.com/privkey.pem
This certificate expires on 2025-08-20.
These files will be updated when the certificate renews.

NEXT STEPS:
- This certificate will not be renewed automatically. Autorenewal of --manual certificates requires the use of an authentication hook script (--manual-auth-hook) but one was not provided. To renew this certificate, repeat this same certbot command before the certificate's expiry date.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
If you like Certbot, please consider supporting our work by:
 * Donating to ISRG / Let's Encrypt:   https://letsencrypt.org/donate
 * Donating to EFF:                    https://eff.org/donate-le
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

到这里，证书就签发完成了，剩下的步骤就是把 `fullchain.pem` 和 `privkey.pem` 这两个文件上传到腾讯云的 SSL 管理后台（为了访问这两个文件，应该还需要用 `sudo` 修改这两个文件所在文件夹的访问权限，建议上传之后把这个文件夹删除，以免出现安全隐患）。

Let's Encrypt 还是非常方便的，只要有域名就可以得到免费的 SSL 证书，而不用给云服务商送钱。不过也是有缺点的，就是在证书到期之前需要记得手动续期，否则网站的访问可能会受到一些影响。
