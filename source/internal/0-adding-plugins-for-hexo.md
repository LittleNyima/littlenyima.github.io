---
title: 开发记录 | 为主题添加乐谱渲染支持
date: 2022-10-05 10:14:23
cover: false
categories:
 - Develop
abcjs: true
---

{% note warning %}

本修改已合并入 butterfly-dev 分支，未来本功能将在主题中集成，不再需要手动修改。

{% endnote %}

{% note flat %}

这是一篇内部开发记录，用于记录为 butterfly 主题添加乐谱渲染支持的开发流程。当后续进行主题版本同步时，可根据此文章快速恢复配置。

{% endnote %}

相关的文件修改：

```text
On branch master
Your branch is up to date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   _config.butterfly.yml
	modified:   themes/butterfly/layout/includes/additional-js.pug
	modified:   themes/butterfly/plugins.yml

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	themes/butterfly/layout/includes/third-party/abcjs.pug
	themes/butterfly/scripts/tag/score.js

no changes added to commit (use "git add" and/or "git commit -a")
```

1. `_config.butterfly.yml`：在主题配置文件中加入 `abcjs` 功能开关。

```yaml
abcjs:
  enable: true
  per_page: false
```

2. `additional-js.pug`：在页面中加入相关 js 脚本。

```stylus
  if theme.abcjs.enable
    if theme.abcjs.per_page
      if is_post() || is_page()
        include ./third-party/abcjs.pug
    else if page.abcjs
        include ./third-party/abcjs.pug
```

3. `plugins.yml`：配置 cdn 地址。

```yaml
abcjs_basic_js:
  name: abcjs
  file: dist/abcjs-basic-min.js
  version: 6.1.9
```

4. `abcjs.pug`：乐谱渲染 trigger 脚本。

```stylus
script.
    function abcjsInit() {
        function abcjsFn() {
            for (let abcContainer of document.getElementsByClassName("abc-music-sheet")) {
                ABCJS.renderAbc(abcContainer, abcContainer.innerHTML, {responsive: 'resize'});
            }
        }
        if (typeof ABCJS === 'object') {
            abcjsFn();
        } else {
            getScript('!{url_for(theme.asset.abcjs_basic_js)}')
                .then(() => {
                    abcjsFn();
                });
        }
    }

    document.addEventListener('DOMContentLoaded', abcjsInit);
```

5. `score.js`：hexo 渲染 tag 过程控制脚本。

```javascript
/**
 * Music Score
 * {% score %}
 */

'use strict';

function score(args, content) {
    function escapeHtmlTags(s) {
        let lookup = {
            '&': "&amp;",
            '"': "&quot;",
            '\'': "&apos;",
            '<': "&lt;",
            '>': "&gt;"
        };
        return s.replace(/[&"'<>]/g, c => lookup[c]);
    }
    return `<div class="abc-music-sheet">${escapeHtmlTags(content)}</div>`;
}

hexo.extend.tag.register('score', score, {ends: true});
```

在启用 `abcjs` 且 `per_page` 选项设置为 `false` 时，若需要在页面中嵌入乐谱，需要在页面的 meta-data 中加入 `abcjs: true`。

# Testcase

```text
X:1
T:alternate heads
M:C
L:1/8
U:n=!style=normal!
K:C treble style=rhythm
"Am" BBBB B2 B>B | "Dm" B2 B/B/B "C" B4 |"Am" B2 nGnB B2 nGnA | "Dm" nDB/B/ nDB/B/ "C" nCB/B/ nCB/B/ |B8| B0 B0 B0 B0 |]
%%text This translates to:
[M:C][K:style=normal]
[A,EAce][A,EAce][A,EAce][A,EAce] [A,EAce]2 [A,EAce]>[A,EAce] |[DAdf]2 [DAdf]/[DAdf]/[DAdf] [CEGce]4 |[A,EAce]2 GA [A,EAce] GA |D[DAdf]/[DAdf]/ D[DAdf]/[DAdf]/ C [CEGce]/[CEGce]/ C[CEGce]/[CEGce]/ |[CEGce]8 | [CEGce]2 [CEGce]2 [CEGce]2 [CEGce]2 |]
GAB2 !style=harmonic![gb]4|GAB2 [K: style=harmonic]gbgb|
[K: style=x]
C/A,/ C/C/E C/zz2|
w:Rock-y did-nt like that
```

{% score %}
X:1
T:alternate heads
M:C
L:1/8
U:n=!style=normal!
K:C treble style=rhythm
"Am" BBBB B2 B>B | "Dm" B2 B/B/B "C" B4 |"Am" B2 nGnB B2 nGnA | "Dm" nDB/B/ nDB/B/ "C" nCB/B/ nCB/B/ |B8| B0 B0 B0 B0 |]
%%text This translates to:
[M:C][K:style=normal]
[A,EAce][A,EAce][A,EAce][A,EAce] [A,EAce]2 [A,EAce]>[A,EAce] |[DAdf]2 [DAdf]/[DAdf]/[DAdf] [CEGce]4 |[A,EAce]2 GA [A,EAce] GA |D[DAdf]/[DAdf]/ D[DAdf]/[DAdf]/ C [CEGce]/[CEGce]/ C[CEGce]/[CEGce]/ |[CEGce]8 | [CEGce]2 [CEGce]2 [CEGce]2 [CEGce]2 |]
GAB2 !style=harmonic![gb]4|GAB2 [K: style=harmonic]gbgb|
[K: style=x]
C/A,/ C/C/E C/zz2|
w:Rock-y did-nt like that
{% endscore %}

# Change Log

2023-04-03：根据 butterfly 作者的建议进行修改。

2023-03-01：支持 `per_page` 配置选项。

2023-02-27：适配到 butterfly-4.7.0 版本，更新了 CDN 配置方法。

2022-10-05：初始版本，实现乐谱渲染基础功能。