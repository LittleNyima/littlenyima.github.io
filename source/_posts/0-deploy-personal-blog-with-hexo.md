---
title: 教程 | Github Pages + hexo 搭建个人博客
date: 2020-10-18 16:41:55
cover: false
categories:
 - Tutorials
tags:
 - Hexo
---
> 注意：这是一篇教程性质的文章，但为了说明的简洁性并不会讲解所有细节。我默认这篇文章的读者具有一定的信息检索能力，并具备各方面的基本知识。

心血来潮决定把个人博客搬迁到自己的网站上，因此采取这种方法进行搭建，并记录一波搭建流程，大概有以下几个步骤：

* 准备一个 Github page 的仓库
* 准备 hexo 框架
* 初始化你的博客
* 部署到服务器端

除此之外，还会提到一些关于博客日常维护和美化的方式。（如果之后研究得比较顺利，或许还可以分享一些定制化、调用第三方 api、性能优化等方面的内容）

# 准备一个Github仓库

一个 Github page 对应于一个名称以 `.github.io` 结尾的仓库，仓库的名称即为 Github page 的地址。与此相关需要做的工作包括：

* 注册一个 Github 账号
* 在本地配置 git
* （可选）配置 git 的用户名、邮箱、配置 ssh 公钥等

上述操作可自行完成，仓库创建好后可以在其中放置一个 `README.md` 文件，再访问对应 Github page 的地址，可以发现 `README` 文件的内容已经在其中显示了，至此 Github page 的准备已经完成。

# 准备hexo框架

hexo 框架的安装需要 Node.js，因此需要先安装 Node.js，可以命令行运行 `npm -v` 验证其是否已经安装成功（正常情况下应当输出一个形如 `6.14.8` 的版本号）。

安装Node.js完成后，使用 `npm install -g hexo-cli` 来安装hexo框架，同样可以在命令行使用 `hexo -v` 来检查其是否安装成功。

# 初始化你的博客

首先在你想要的位置创建一个文件夹，然后将工作目录切换到这个文件夹（windows下可以在右键菜单选择“Git Bash Here”，其他系统可能要经过一系列 `cd` 操作）。

工作目录切换完成后，工作目录所在的文件夹就将成为你的博客所在的文件夹。使用 `hexo init` 来把这个文件夹初始化成一个博客文件夹，然后使用 `hexo g` 来生成网站，因为你还没有创建博客文章，因此其中会自带一篇 Hello World 的文章。

如果你想要观察网站的效果，可以使用 `hexo s` 将生成的网站挂载到本地服务器，然后在浏览器访问 [localhost:4000](localhost:4000)，就可以看到效果了。

# 将博客部署到服务器端

当你对自己博客的效果满意后，便可以部署到服务器端。

因为此处要部署到 Github page，因此以部署到 Github 为例。首先需要安装 hexo-deployer-git 模块，在博客文件夹目录下，运行 `npm install hexo-deployer-git --save` 来进行安装。

然后修改博客的配置文件，在目录中有一个 `_config.yml`，用适当的文本编辑器打开，找到其中的 `# Deployment` 一项，将其修改为形如以下形式：

```yaml
# Deployment
## Docs: https://hexo.io/docs/one-command-deployment
deploy:
  type: git
  repository: git@github.com:LittleNyima/littlenyima.github.io.git
  branch: master
```

其中 `repository` 一项中填自己仓库的 git 地址，这个地址可以在 Github 的仓库页面找到，如果没有配置 ssh 公钥，应当使用 `https://` 开头的地址，否则推荐使用 `git@` 开头的地址。

`branch` 一项中填写希望推送到的分支，如果你对 git 不够了解，可以直接填入 ~~`master`~~ `main`（更新：在无特殊设置的情况下，Github 的默认分支名已经从 `master` 变为 `main`）。

> 修改这一项的同时，也可以顺便修改一下配置文件中的其他项，例如站点名称、作者等信息。

在以上的一切都配置妥当后，使用 `hexo d` 命令将其部署到服务器端，部署完成后再访问对应 Github page 页面，应当可以看到你的博客页面，至此，个人博客已经部署成功。

> 注意：之后每次对博客进行修改后，在部署前都应当先使用 `hexo g` 来更新文件。

如果想要添加博客，可以使用命令行的 `hexo new` 命令，也可以直接在 `source/_posts/` 创建新的文件。

# 博客的美化——使用主题

hexo 框架可以使用一系列主题对博客进行美化，访问 [https://hexo.io/themes/](https://hexo.io/themes/) 来获取主题。一般来说各种主题都会随附比较详细的使用文档，这里以我个人使用的 butterfly 主题（3.2.0版本）为例讲解主题的使用方式。

首先获取主题，主题的获取方式一般可以从 git 仓库进行 clone，clone 到 `themes/` 文件夹下。例如：

```shell
git clone -b master https://github.com/jerryc127/hexo-theme-butterfly.git themes/butterfly
```

然后修改 `_config.yml` 文件，修改 `theme` 的取值：

> theme: butterfly

然后安装所需插件：`npm install hexo-renderer-pug hexo-renderer-stylus`。此时重新生成文件并挂载，可以发现主题发生了变化。

butterfly 主题提供了比较方便且详尽的定制化方法，具体可以参阅其[文档](https://demo.jerryc.me/posts/21cfbf15/)。下面讲解一些比较经常用到的定制化方法。

在 `themes/butterfly/` 文件夹下同样包含一个 `_config.yml` 文件，将其复制到博客的根目录下并重命名为 `_config.butterfly.yml`，主题的大多数内容可以在其中修改。这个新的配置文件具有较高优先级，因此当其中内容与基础配置文件冲突时，会发生覆盖。

在这个配置文件中同样具有比较详尽的注释与使用方式示例，常用的有：

* 菜单栏的设置：在 `menu` 项中加入条目，例如要在菜单栏加入文字为“首頁”，超链接地址为 `/`，图标为 home 图标的项目，可以加入 `首頁: / || fas fa-home` 的条目。

    > `fas fa-home` 是 Font-Awesome 图标在 yaml 文件中的使用形式。如果需要使用 Font-Awesome 图标，可以访问 [https://fontawesome.com/icons](https://fontawesome.com/icons)，在其中找到所需的图标并使用。

    注意，加入条目后为了防止链接到不存在的页面，需要创建相应的页面。以标签页为例，使用 `hexo new page tags` 创建新的页面，然后修改 `source/tags/index.md` 文件头部的配置信息，例如：

    ```yaml
    title: 標籤
    date: 2020-10-18 18:18:06
    type: "tags"
    ```

    这部分配置信息的语法规则与 yaml 的语法规则相同，对于一些特定页面，需要设置 type。

* 代码风格设置：包括代码块主题、复制按钮、显示语言、支持收起、自动换行等项目。

* 社交图标设置：与菜单栏设置方法类似，在 `social` 条目下加入项目，例如 `fab fa-github: https://github.com/LittleNyima || Github`，便可在名片底部插入一个社交图标。

* 网站各部分图片设置：包括网站图标、头像、网站头图、背景图等内容。

* 其他项目的设置：请参阅文件内容与注释。

有关使用 hexo 配置博客的基本知识大概就是这些，当然也有一些潜在的问题，例如如何备份博客环境、使用中文标签会导致 url 中出现中文字符等内容，仍有待解决，如果找得到解决方案的话或许会在以后的博客里继续更新。（发出咕咕咕的声音）

最后，如果大家对我的博客感兴趣，欢迎持续关注！