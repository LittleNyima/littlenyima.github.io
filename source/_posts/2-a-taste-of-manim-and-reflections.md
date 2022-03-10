---
title: 技术相关 | Python 动画引擎 manim 浅度体验与思考
date: 2022-03-01 16:36:18
cover: false
categories:
 - Techniques
tags:
 - Manimgl
 - Animation engine
---

前段时间在写 [Wordle Solver](https://github.com/LittleNyima/wordle-solver)，查阅资料的时候发现了 [3Blue1Brown 的 Youtube 频道](https://www.youtube.com/c/3blue1brown)，在看视频的时候发现频道中的视频基本上都是用一个叫做 [manim](https://github.com/3b1b/manim) 的动画引擎做的。我感觉蛮有意思，就安装下来简单体验了一下。

manim 最初是由一位个人开发者开发的，由于其不承诺长期提供支持，后来又出现了 3b1b 维护的 manimgl 版和 manimCE 社区版。这三种的特性与具体用法都存在一定的差异，考虑到 manimgl 版本可以使用 OpenGL 进行渲染，并且已经出现了一些基于它制作的比较高质量的成品视频，我在体验时选择了 manimgl 版。

# manim 浅度体验

manimgl 使用 pip 即可安装：

```shell
pip install manimgl
pip install pyopengl
```

除安装上述包外，还需要安装 ffmpeg 用于视频编解码，以及 LaTeX 以支持公式渲染（推荐使用 Tex Live 发行版，功能相对强大一点）。由于这两者我的开发环境中已提前配置，所以无需重复这一步骤。

配置完成后，即可运行其自带的 demo 脚本：

```shell
manimgl example_scenes.py OpeningManimExample
```

运行后，会出现两个窗口，其中一个是 IPython 的交互式命令行窗口，另一个是播放动画画面的图形窗口。在动画播放时，命令行窗口中会显示渲染进度。动画播放结束后，会进入交互模式。在交互模式中，可以使用滚轮使图像上下平移、按住 `Z` 键滑动滚轮进行缩放、按住 `D` 键拖动鼠标进行模视变换等。

{% note info flat %}

在部分情况下，由于显卡性能问题，会出现画面渲染不完整的情况。此时应当查看系统设置，修改 Python 使用的显卡为较高性能的显卡。

{% endnote %}

{% raw %}

<style>
    .resp-container {
        position: relative;
        overflow: hidden;
        padding-top: 65%;
        margin-bottom: 16px;
    }
    .resp-iframe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border: 0;
    }
</style><div class="resp-container"><iframe class="resp-iframe" src="https://player.bilibili.com/player.html?aid=679492789&bvid=BV12S4y1g7pm&cid=530012902&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe></div>

{% endraw%}

从例子中不难发现，manim 对数学相关元素，例如文字、公式、图形、坐标轴等元素都有很好的支持，能够创建一些补间动画，并且这些动画支持某些特定的动画曲线。

阅读 `example_scenes.py` 的代码，可以猜测，在 manim 中，所有的动画都是 `Scene` 的子类，在运行一个脚本时，参数对应的子类将被实例化，并通过调用 `construct()` 方法装载并渲染。以上述视频对应的代码为例：

```python
class OpeningManimExample(Scene):
    def construct(self):
        intro_words = Text("""
            The original motivation for manim was to
            better illustrate mathematical functions
            as transformations.
        """)
        intro_words.to_edge(UP)

        self.play(Write(intro_words))
        self.wait(2)

        # Linear transform
        grid = NumberPlane((-10, 10), (-5, 5))
        matrix = [[1, 1], [0, 1]]
        linear_transform_words = VGroup(
            Text("This is what the matrix"),
            IntegerMatrix(matrix, include_background_rectangle=True),
            Text("looks like")
        )
        linear_transform_words.arrange(RIGHT)
        linear_transform_words.to_edge(UP)
        linear_transform_words.set_stroke(BLACK, 10, background=True)

        self.play(
            ShowCreation(grid),
            FadeTransform(intro_words, linear_transform_words)
        )
        self.wait()
        self.play(grid.animate.apply_matrix(matrix), run_time=3)
        self.wait()

        # Complex map
        c_grid = ComplexPlane()
        moving_c_grid = c_grid.copy()
        moving_c_grid.prepare_for_nonlinear_transform()
        c_grid.set_stroke(BLUE_E, 1)
        c_grid.add_coordinate_labels(font_size=24)
        complex_map_words = TexText("""
            Or thinking of the plane as $\\mathds{C}$,\\\\
            this is the map $z \\rightarrow z^2$
        """)
        complex_map_words.to_corner(UR)
        complex_map_words.set_stroke(BLACK, 5, background=True)

        self.play(
            FadeOut(grid),
            Write(c_grid, run_time=3),
            FadeIn(moving_c_grid),
            FadeTransform(linear_transform_words, complex_map_words),
        )
        self.wait()
        self.play(
            moving_c_grid.animate.apply_complex_function(lambda z: z**2),
            run_time=6,
        )
        self.wait(2)
```

从上面的代码可以看出，在创建一个动画时，用户并不需要为每一个对象的风格以及其运动的具体方式指定复杂的参数，而是只提供最基本的元素与变化方式，以及一些简单的外观属性（例如文字的字号、颜色、位置等）。

# 一些相关的思考

作为一个面向数学元素可视化的动画引擎，manim 提供了一套易于开发与拓展的框架。在使用时，用户无需过多考虑具体的参数，即可得到外观相当不错的视频。对于这个框架，我也有一些自己的看法：

- 作为一套面向数学的框架，manim 支持的大多是已经确定的元素，并且由于缺少对 mesh 的支持，比较难以创建自定义的元素。（当然，这也与其本身的定位有关。）
- 为了创建动画，创作者必须掌握 Python 语言。我认为基于 config 的做法比基于代码实现的做法对大多数创作者更友好一些。（虽然基于 config 也会引入相应的问题，例如在动画生成过程中需要进行数值计算，或想要为某个动画应用一个自定义的动画曲线时，config 无法处理所需的计算。）
- 除了代码本身之外，要想使用该框架，还需要安装额外的软件（如 ffmpeg、LaTeX 等），增加了入门成本。我认为可以将这些额外的功能（如视频编码、公式支持等）作为插件引入框架，对于无需导出视频或使用公式的用户来说，则无需安装这些额外功能。

当然对于有意于使用该框架的用户来说，想必上述问题都不难解决；对于无法解决上述问题的用户来说，也有替代方案（例如直接使用 Adobe After Effects）。不过对于一个框架或软件来说，提高可用性、降低入门成本还是非常重要的。

总结来说，这个引擎还是非常有趣的，在调研的过程中我也发现了不少用该引擎制作的质量不低的视频。一直以来我都对多媒体创作相关的内容比较感兴趣，如果有精力的话或许以后可以在 manim 的基础上增加一些扩展，或开发一套符合自己设计理念的动画引擎。