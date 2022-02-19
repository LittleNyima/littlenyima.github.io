---
title: 教程 | 利用 Github Actions 实现代码推送后自动部署
date: 2022-02-18 20:09:21
cover: false
tags:
 - Tutorials
---

博客自从搭建以来已经搁置一段时间了，最近想在友链里多加上几个好友的链接，但在我将最新的页面代码推送到远程仓库后，Github Pages 的部署流程并没有被成功触发。因此，我决定研究一下 Github Actions 的用法，并在此记录一下。

[GitHub Actions](https://docs.github.com/en/actions) 可以自动化地实现一些工作流，例如在代码推送或 pull request 发起时，进行一系列测试、打包、部署等操作。当工作流启动时，它会创建一个容器，并安装软件、配置环境，在流程结束后还会将生成的数据推送到指定的位置。

我的博客使用 hexo 框架进行搭建，版本控制采取“开发-部署”的双分支模型。因此，在我需要更新我的博客时，首先我需要在博客的源代码中进行编辑，然后依次运行 `hexo clean`、`hexo server`、`hexo deploy` 进行预览和部署，然后将源代码 commit 并推送到远程，进行版本控制。这一流程可以使用 Github 的 Actions 功能进行简化。利用 Github Action，在代码推送后，Github 服务器可以自动启动编译与部署的流程。

# 仓库读写权限配置

为了使用 Github Action 实现部署，需要进行预先准备。首先，需要创建一个 ssh 公钥，用于工作流容器对仓库的访问。具体流程为：

- 在本地使用 `ssh-keygen -f gh-pages-deploy` 生成 ssh 密钥。
- 生成后，在本地工作目录下应当出现 `gh-pages-deploy` 和 `gh-pages-deploy.pub` 一对私钥和公钥。
- 打开 Github 仓库的页面，在源代码仓库的 `Settings > Secrets > Actions` 一栏中加入私钥。
- 在部署用仓库的 `Settings > Deploy keys` 一栏中加入公钥（同时应勾选“Allow write access”选项）。

{% note warning flat %}

拥有 ssh 密钥的人可以获取你的仓库的读写权限，请妥善保存，避免泄露！

{% endnote %}

# 编写工作流

在仓库的 `Actions` 选项卡中点击 `New workflow` 可以创建新的工作流。在创建时，可以直接使用现有的工作流（在 Github Marketplace 中可以获取一系列编写好的工作流），也可以编写自己的工作流。

在源代码仓库的根目录中创建 `.github/workflows/deploy-on-push.yml`，并向其中加入以下内容：

```yaml
# This is a basic workflow to help you get started with Actions

name: deploy-on-push

# Controls when the workflow will run
on:
  # Triggers the workflow on push or pull request events but only for the master branch
  push:
    branches: [ master ]
  # pull_request:
    # branches: [ master ]

  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:

# Environment variables
env:
  GIT_USER: LittleNyima
  GIT_EMAIL: littlenyima@163.com
  # THEME_REPO: jerryc127/hexo-theme-butterfly
  # THEME_BRANCH: master
  DEPLOY_REPO: LittleNyima/littlenyima.github.io
  DEPLOY_BRANCH: deploy

# A workflow run is made up of one or more jobs that can run sequentially or in parallel
jobs:
  # This workflow contains a single job called "build"
  build:
    name: Build on node ${{ matrix.node_version }} and ${{ matrix.os }}
    # The type of runner that the job will run on
    runs-on: ubuntu-latest
    strategy:
      matrix:
        os: [ubuntu-latest]
        node_version: [12.x]

    # Steps represent a sequence of tasks that will be executed as part of the job
    steps:
      # Checks-out your repository under $GITHUB_WORKSPACE, so your job can access it
      - name: Checkout
        uses: actions/checkout@v2
      
      # - name: Checkout theme repository
      #   uses: actions/checkout@v2
      #   with:
      #     repository: ${{ env.THEME_REPO }}
      #     ref: ${{ env.THEME_BRANCH }}
      #     path: themes/butterfly
      
      - name: Checkout deploy repository
        uses: actions/checkout@v2
        with:
          repository: ${{ env.DEPLOY_REPO }}
          ref: ${{ env.DEPLOY_BRANCH }}
          path: .deploy_git
      
      - name: Use node.js ${{ matrix.node_version }}
        uses: actions/setup-node@v1
        with:
          node-version: ${{ matrix.node_version }}
          
      - name: Configure environment
        env:
          DEPLOY_SECRET: ${{ secrets.GH_PAGES_DEPLOY_SECRET }}
        run: |
          sudo timedatectl set-timezone "Asia/Shanghai"
          mkdir -p ~/.ssh/
          echo "$DEPLOY_SECRET" > ~/.ssh/id_rsa
          chmod 600 ~/.ssh/id_rsa
          ssh-keyscan github.com >> ~/.ssh/known_hosts
          git config --global user.name $GIT_USER
          git config --global user.email $GIT_EMAIL
          # cp _config.theme.yml themes/butterfly/_config.yml
      
      - name: Install dependencies
        run: npm install

      - name: Deploy pages
        run: npm run deploy
      
      # Runs a single command using the runners shell
      # - name: Run a one-line script
      #   run: echo Hello, world!

      # Runs a set of commands using the runners shell
      # - name: Run a multi-line script
      #   run: |
      #     echo Add other actions to build,
      #     echo test, and deploy your project.
```

其中，`env` 中的各项环境变量需要根据实际情况修改，对于第 `20` 到 `21` 行和 `43` 到 `48` 行被注释的部分，如果你的主题文件夹是作为 git submodule 保存的，则需要取消注释，以实现主题代码的拉取。我为了保持所使用主题版本的稳定性，以达到比较好的兼容性，直接将主题文件夹以静态资源的方式保存在了 `themes` 文件夹下，因此无需拉取相应代码。

在以上的 yaml 文件中，各项的含义分别为：

- name：表示工作流 / Job / Step 的名称。一个工作流由数个 Job 组成，这些 Job 可以以串行或并行的方式运行，每个 Job 又由一系列 Step 组成。`name: deploy-on-push` 表示工作流的名称为 `deploy-on-push`。 
- on：表示工作流被触发的条件，例如 `on.push.branches: [master]` 表示对 master 分支推送代码时触发该工作流。
- env：由一系列键值对组成，用户可以在这里定义一些环境变量，并用 `${{ var }}$` 的形式引用。在这个工作流中，各个环境变量的含义分别为：
  - env.GIT_USER：编译后部署使用的 git 用户名，这个名字会显示在部署分支的 commit 记录中。
  - env.GIT_EMAIL：编译后部署使用的 git 邮箱，同理，该邮箱也会在 commit 记录中出现。
  - env.THEME_REPO：hexo 主题所在仓库，例如我使用的主题为 `jerryc127/hexo-theme-butterfly`。
  - env.THEME_BRANCH：hexo 主题所在分支，拉取主题代码时仅会拉取该分支的代码。
  - env.DEPLOY_REPO：hexo 编译后要部署到的仓库，一般来说是你的 `username.github.io` 仓库。
  - env.DEPLOY_BRANCH：hexo编译后要部署到的分支，这应当与你 Github Pages 使用的分支一致。
- jobs：工作流含有的 Job，每个 Job 是一个复合结构，其中定义了运行环境、运行步骤等内容。
  - jobs.{job}.runs-on：表示 Job 运行所需的平台环境，例如`ubuntu-latest`、`windows-latest`、`macos-latest`。
  - jobs.{job}.steps：Job 所含有的工作步骤。
    - jobs.{job}.steps.name：步骤名，编译时会以 LOG 形式输出。
    - jobs.{job}.steps.uses：所要调用的 Action。其他用户封装了一些现成的 Action，可以像函数一样调用，实现一些功能。例如上述模板中所使用的 `actions/checkout@v2` 可以用于检出 Github 仓库的代码、`actions/setup-node@v1` 可以用于安装 node.js 环境等。
    - jobs.{job}.steps.with：调用 Action 传的参数，格式与具体 Action 有关。

Github Action 模板的具体语法可以参考 [Github 文档](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)。如果不想自己研究，直接修改 `env` 下的内容即可。

如此配置完成后，将 yaml 文件 commit 到源代码所在仓库，push 后便可以触发编译部署的工作流。

> 参考资料：[利用 Github Actions 自动部署 Hexo 博客 | Sanonz](https://sanonz.github.io/2020/deploy-a-hexo-blog-from-github-actions/)

