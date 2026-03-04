---
title: 笔记｜从 Prompt 到智能体：LLM Agent 的演进逻辑与工程实践全解析
date: 2026-03-04 18:26:05
cover: false
mathjax: false
categories:
 - Notes
tags:
 - Generative models
 - Large language models
 - Agents
---

随着前段时间 OpenClaw 的爆火，各种 agent 又掀起了一阵热潮，因此在这里系统性地总结一下有关 agent 的一些知识。

# LLM 应用编年史

在开始介绍之前我们不妨回顾一下 LLM 应用的发展史，最初出现的 ChatGPT 其形式非常简单直接，也就是一个对话窗口，我问你答。在这个阶段，用户使用 LLM 的方式就是比较简单的问答，并且还有一些基于 in-context learning 的对话策略，也就是在对话时先给出几个例子，让 LLM 仿照例子进行输出。比如下面的这个 prompt：

```
// User:
Classify the sentiment of the following text as positive, negative, or neutral.
Text: The product is terrible. Sentiment: Negative
Text: Super helpful, worth it Sentiment: Positive
Text: It doesnt work! Sentiment:

// Assistant:
Negative
```

后来人们很快就发现了大模型的潜力，在这个时期爆发出了非常多的各种用途的 prompt，在使用时首先把 prompt 复制给大模型，就可以让大模型发挥某种角色的作用。甚至当时出现了「提示词工程师」这样一种职业，虽然当时就感觉有点扯，但是从就业市场来看确实有不少人为此买账。

不管用法如何，可以看到在这个阶段 LLM 的应用主要依然停留在调节提示词这个层级。同时在这个时期，除了模型能力本身之外，LLM 有一个非常大的痛点，也就是 Knowledge Cutoff。由于大模型的训练数据仅包含某个时间点之前的信息，因此 LLM 无法「感知现在」，很多回答缺乏时效性。为了解决这个问题，首先出现的技术是 RAG，也就是检索增强生成。

这个技术最初的实现是非常简单的，因为大模型无法得知近期发生的事情，那么一个非常直观的解决方案就是直接拿着关键词去 Google 搜索，然后把搜索结果放到上下文中，这样大模型就能够知道最新的信息，而不受训练数据的影响。后来随着 LLM 应用工程化的发展，出现了 LangChain 这样的非常完备的工程化库，其实现了一套完备的 RAG 流程，也就是：

```
预处理阶段：加载文档 -> 文档切片 -> embedding -> 存储到向量数据库
推理阶段：prompt embedding -> 向量数据库匹配 -> 加入上下文 -> 生成
```

随着 RAG 一同出现的是 Tool Call，这个思路还是很直接的：既然我可以调用外部的搜索引擎或者知识库来协助生成过程，那调用其他工具是不是也可以？理论上来说是的，只要我们能够找到一种方式来判断「大模型是否需要外部工具协助」，就可以适时地调用外部工具并且把结果放到上下文中，从而完成外部工具对生成过程的赋能。

虽然道理是这样，但是在早期使用时，工具调用的效果并不是很好。总结下来，影响工具调用效果的因素主要有两个：其一是何时应该调用工具以及调用什么工具，其二是使用什么参数来调用工具。

在 Tool Call 刚刚出现时，工具调用基本上是纯靠提示词工程实现的，也就是通过提示词来告诉大模型用一种特定的模式来表示工具调用，再从输出中搜索这个模式，从而识别出应该调用的工具。以经典的 agent 模式 ReAct 为例：

```
// User:
周杰伦现在的年龄加上 10 是多少？

// Assistant:
Thought 1: 我需要先知道周杰伦的出生日期，然后计算他现在的年龄。
Action 1: Search["周杰伦 出生日期"]
Observation 1: “周杰伦，1979年1月18日出生于台湾。”
Thought 2: 他出生于 1979 年，现在是 2024 年（假设当前时间），所以他是 45 岁。我需要再加 10。
Action 2: Calculator[45 + 10]
Observation 2: 55
Thought 3: 我已经得到了最终结果。
Final Answer: 周杰伦现在的年龄加上 10 是 55 岁。
```

可以看到在这个阶段，Action 是用一个类似 Tool[Arguments] 的格式来表示的，受限于当时大模型的性能，如果在生成工具调用时格式不准确，很有可能导致工具调用失败。除此之外这个 pattern 也很容易和其他的正常内容混淆，这种工具调用的实现方式更多的是对模型本身能力的妥协。

从现在的视角来看，工具调用对 LLM 来说并不是一个天生就会的操作，后来的很多模型都专门针对工具调用做了微调，甚至在某些模型中，工具调用操作直接被加入了 special token 中。例如最近新发布的 Qwen 3.5 的词表中，tool_call 就赫然在列：

```json
// Huggingface Qwen/Qwen3.5-397B-A17B/tokenizer_config.json
"248058": {
    "content": "<tool_call>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false,
    "special": false
},
"248059": {
    "content": "</tool_call>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false,
    "special": false
}
```

通过这样的针对性训练，LLM 可以原生支持工具调用，使工具调用成为「一等公民」。

实际上可以看到，自从工具调用的能力被赋予 LLM，大模型的应用就不仅仅局限在对话中，agent 也就从此出现了。从我个人的理解而言，agent 本身就是 LLM+工具这两者的结合，不管实现方式是简单粗暴的对话 SDK+手动调用外部函数，还是像 LangGraph 这样用状态机来维护系统和外部的交互逻辑，归根结底，agent 就是一个 LLM 加上其可以调用的工具。

如果只是为了介绍 agent，编年史的部分到这里就算是可以结束了。不过既然介绍到这里了，不如就一口气把后来出现的 MCP 和 Skills 也讲一下。这两个概念虽然说感觉起来很 fancy，但是依然没有脱离上面讲过的 LLM+工具这个框架。

首先我们来总结一下 Tool Call 的过程，根据我个人的理解，在推理时：

1. 用户在调用 LLM 时给出工具的定义，类似下边这样：
    
    ```
    client.responses.create(input=prompt, tools=tools, ...)
    ```
    
2. 模型侧在接收到工具定义后，将其序列化并加入到上下文中的某个位置（例如序列化成一个 JSON 字符串，放到 system prompt 和 user prompt 之间）；
3. 由于模型在训练阶段已经学会了理解工具的定义，在推理过程中，当模型认为需要调用工具时，就输出表示工具调用的 token（类似上边给出的 Qwen 3.5 的 special token）；
4. 模型的服务端在 response 中发现类似 `<tool_call>{...}</tool_call>` 的序列后，就返回一个信号，类似 `{type: "function_call"}` ，并把命中的工具以及调用参数随之发送；
5. 客户端收到 response 后，发现要进行工具调用，就使用返回的参数真实地调用工具，并把工具调用的结果再次发送给 LLM；
6. LLM 收到工具调用的结果并继续生成，循环此过程可以进行多轮工具调用。

回看上述的过程，可以发现整个流程都是有比较成熟的定义的，无需用户手动维护，而其中用户主要需要提供的两个部分，一是工具本身的定义，二是用来描述任务执行逻辑的 prompt。

在上述的两个部分中，MCP 主要简化的是工具的定义方式。在传统的工具调用过程中，工具的定义与大模型的调用是耦合在一起的。也就是说，当我想要让 LLM 调用某个工具时，我必须在我的项目中把这个工具作为一个函数或者一个类实现出来，并且作为参数传给 LLM client，才能被正确地调用。对于一些比较复杂、甚至是编程语言或运行环境与项目本身不适配的工具，要想将其引入，无疑会增加项目复杂度和开发负担。

MCP 正是为了解决上边所说的这个问题而被提出的。在 MCP 协议中，工具被定义在模型调用程序外部的某个服务器上，工具不再需要被引入到项目中，而是能够通过类似 RPC 的方式被 LLM 远程调用。举一个简单的例子，例如我希望用 LLM 自动帮我回复知乎和小红书的评论，那么如果使用传统的工具调用方式，我就需要把有关这两个平台的代码、密钥等都集成进我的项目中。尤其是多个项目都使用同一套逻辑时，需要把这套逻辑适配到每一个项目中。

而有了 MCP 之后，工具的开发者可以将工具打包进一个 MCP Server 提供给使用者，工具本身的定义以及其使用的提示词、其他资源等都可以通过这个 Server 进行标准化地访问。LLM 只需要知道这个 MCP Server 的地址，就能够通过一个 MCP Client 直接调用其提供的工具。让我们再次梳理一下相比传统的 Tool Call，使用 MCP 调用工具时都发生了什么：

1. 首先在应用启动阶段，在创建 LLM 客户端时，同时还创建一个 MCP 客户端。这个客户端根据所收到的 MCP Server 的地址，向 MCP Server 请求一系列工具以及资源的定义；
2. 从这一步开始则是和传统的 Tool Call 大同小异：用户在调用 LLM 时给出 MCP 工具的定义；
3. 模型侧在接收到工具定义后，将其序列化并加入到上下文中的某个位置（例如序列化成一个 JSON 字符串，放到 system prompt 和 user prompt 之间）；
4. 由于模型在训练阶段已经学会了理解工具的定义，在推理过程中，当模型认为需要调用工具时，就输出表示工具调用的 token（类似上边给出的 Qwen 3.5 的 special token）；
5. 客户端收到 response 后，发现要进行工具调用，就通过 MCP Client 向对应的 MCP Server 发起工具调用的请求，MCP Server 完成工具调用后将结果返回给 MCP Client，客户端再将 MCP Client 收到的工具调用的结果再次发送给 LLM；
6. LLM 收到工具调用的结果并继续生成，循环此过程可以进行多轮工具调用。

可以看到工具调用的内核是不变的，只是在 LLM 和工具之间添加了一层标准化的协议，从而简化了工具集成的难度。实际上，把工具进行远程部署或者通过网络请求等方式调用工具并不是什么新鲜事，但 MCP 把这件平常的事实现了标准化，这对于整个社区的构建是非常重要的。

在我们刚刚提到的两个问题中，MCP 解决了工具定义的问题，而 Skills 则是完成了 prompt 的标准化与结构化管理。就像我们上边说过的一样，当我们要求 LLM 完成某项任务时，通常会将一套完整的 prompt 引入上下文中。这个过程在最开始是通过手动复制粘贴完成的，后来也出现了一些平台可以为 LLM 设置「人设」，也提供了一种 prompt 复用的方式。就近两年的 AI 编程产品来说，cursor 的 cursorrules 文件以及 claude code 的 CLAUDE 文件也能够支持注入一些 prompt。

随着 agent 需求和功能的发展，用户使用的 prompt 也日渐复杂。例如在 AI 编程领域，用户高频使用的场景可能同时覆盖项目开发、测试以及版本管理等，这里的每个场景可能都对应一套 prompt，例如定义了项目代码规范的 SKILL、定义测试流程的 SKILL，以及定义 git commit message 格式的 SKILL 等等。除此之外，每套 prompt 都对应特定的垂直场景，并不会全程都使用到。因此在 agent 使用过程中需要灵活地调度这些 prompt，而 SKILLS 可以比较好地解决这个问题

从本质上来说，SKILL 是一个结构化组织的 prompt 库，每一个 SKILL 都提供了一些元信息，例如 SKILL 的描述。在调用 LLM 时，和工具调用类似，在开始时，现有的各个 SKILL 的描述会被提供给 LLM，模型会从中匹配到需要使用的 SKILL，并将其内容注入到当前的上下文中。通过这种方式，即可完成 prompt 的动态挂载。

作为本小节的总结，我们不难发现现有的各种 agent，其本质都是 LLM+外部工具的组合。同时，现有的 AI 应用的核心都是使用各种工程化的手段来有效地管理大模型的上下文，使其高效地发挥能力，从而影响外部世界。因此我们的下一个议题就是讨论一下 agent 开发时常用的工程策略。

# Agent 应用的工程实践

经过上边的讨论，我们已经对 agent 应用发展过程中的一些主流技术有了比较完善的了解。然而，虽然已经有了 MCP、SKILLS 这种能够大大简化 agent 构建的技术，但很明显光靠这些我们是无法实现出像 OpenClaw 或者 cursor 这种实用的 agent 应用的。除了应用开发本身需要的技术之外，实现一个实用的 agent 应用通常需要大量的工程优化，使 LLM 的能力能够被充分运用。

在 agent 构建的过程中，上下文管理是核心中的核心，其不仅关系到推理成本，也决定了 LLM 响应的准确性。因此这里我们以上下文管理策略为例直观地感受一下 agent 应用在工程化时的一些主流做法。所谓的上下文管理，也就是通过控制 LLM 的输入来为 LLM 提供尽可能精确、充分的信息。不同 agent 应用的上下文管理策略不一而足，这里主要总结一下不同工具的上下文管理策略。

## Manus

在 [Manus 的文章](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)中，首先被讨论的一个点是 **KV-cache**。Manus 关于 agent 的一个重要的发现是，在 agent 使用 LLM 的过程中，LLM 的输入 token 数是显著大于输出 token 数的。这一点非常容易理解，因为上下文中主要的信息通常都来自于外部的文档或代码，而针对这些信息的指令通常都是总结或进行结构化地修改。Manus 给出了 prefilling 和 decoding 所占用 token 数的比例，即 100:1。在这种情况下，输入 token 能否有效利用缓存在极大的程度上决定了整个推理的成本。因此，为了有效利用缓存机制，保持上下文的前缀稳定非常重要。

另一个非常关键的策略是 Manus 的上下文压缩策略，在各种上下文压缩策略中，Manus 的压缩策略属于比较保守的一种。实际上 Manus 并不真正地压缩上下文的内容，而是通过将上下文中的文档外部化到文件系统中缩减上下文的大小。也就是说，当上下文达到一定的长度后，Manus 将历史中的部分文档持久化到文件系统中，而在上下文中仅保留必要的引用和描述。Manus 这样做的出发点在于，任何不可逆的压缩都是带有风险的，某些内容虽然在当前看来并不重要，但它有可能对于未来又是关键性的信息。所以 Manus 把文件系统作为终极的上下文：大小不受限制，天然持久化，并且代理可以直接操作。

为了解决长上下文中主题和早期目标丢失的问题，Manus 的策略是维护一个 TODO List，并在执行任务的过程中对其不断进行更新，从而让任务目标不断地出现在更新的上下文中，从而得到保持。（这也是现在很多工具中 Plan Mode 的做法）

## Claude Code

在有关 Claude Code 的[官方 blog](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) 以及一些[外部的分析](https://www.southbridge.ai/blog/claude-code-an-analysis)中，我认为比较值得注意的主要有两个要点，其一是其上下文压缩策略，其二是基于 sub-agent 进行上下文隔离。

在上下文压缩时，Anthropic 提到了其对内容压缩的取舍，被压缩的内容主要包括重复的工具调用结果以及重复的 message，而保留的内容包括架构层面的决策、未解决的 bug 以及实现细节。从 [Claude Code 的 System Prompt](https://github.com/Piebald-AI/claude-code-system-prompts/blob/main/system-prompts/agent-prompt-conversation-summarization.md) 中可以找到关于上下文压缩的要求描述，主要包含以下内容：

```
1. Primary Request and Intent - 用户提出的所有明确请求以及其背后的核心意图
2. Key Technical Concepts - 对话中涉及的所有重要技术概念、具体技术栈、框架及其应用
3. Files and Code Sections - 所有查看过、修改过或新创建的文件及代码部分
4. Errors and fixes - 过程中遇到的所有错误及相应的解决方案
5. Problem Solving - 已解决的问题以及目前正在进行的故障排除和调试工作
6. All user messages - 所有由用户发送的消息（不包括工具生成的运行结果）
7. Pending Tasks - 用户明确要求完成但目前尚未处理的挂起任务清单
8. Current Work - 在本次总结请求之前正在进行的具体工作
9. Optional Next Step - 与当前工作直接相关的下一步计划
```

另外一个关于上下文压缩的发现是，在我对 Claude Code 的网络请求进行抓包分析时，发现在发送消息之前 Claude Code 会进行一个「是否是新话题」的询问，内容如下：

```json
{
  "system": [
    {
      "type": "text",
      "text": "x-anthropic-billing-header: cc_version=2.1.15.d11; cc_entrypoint=cli"
    },
    {
      "type": "text",
      "text": "You are Claude Code, Anthropic's official CLI for Claude."
    },
    {
      "type": "text",
      "text": "Analyze if this message indicates a new conversation topic. If it does, extract a 2-3 word title that captures the new topic. Format your response as a JSON object with two fields: 'isNewTopic' (boolean) and 'title' (string, or null if isNewTopic is false). Only include these fields, no other text. ONLY generate the JSON object, no other text (eg. no markdown)."
    }
  ]
}
```

或许这个是否是新话题的判断也会影响 Claude Code 的上下文管理策略，例如其可能会对旧的话题的内容进行截断。但由于现有的分析中对此的讨论不多，这里暂时不展开讨论。

Anthropic 还提到了使用 sub-agent 架构来管理上下文，也就是说在顶层有一个主要的 agent 负责统筹全局的规划，而在执行具体任务时，会使用一个带有单独上下文的 agent。对于顶层 agent 来说，执行任务的具体细节是透明的，其接收到的仅有 sub-agent 的输入以及总结性质的输出。（SouthBridge 在 *[Conducting smarter intelligences than me: new orchestras
](https://www.southbridge.ai/blog/conducting-smarter-intelligences-than-me-new-orchestras)* 这篇文章中应用了这种架构对复杂的代码工程进行了分析，并详细讨论了这种方法与其他几种 agent 的组织模式之间的比较，有兴趣的读者可以自行阅读一下）

除了上述两点之外，Claude Code 也使用了我们在 Manus 的章节中提到的 note-taking 策略。具体使用哪种策略在很大程度上也取决于任务的性质，例如：

1. 对于需要在上下文中前后跳跃的任务，使用上下文压缩的策略更合适；
2. 对于具有明确目标以及里程碑的任务，使用 note-taking 策略更合适；
3. 对于需要复杂的研究和分析，或者需要并行执行任务时，sub-agent 策略更合适。

## 小结

从上面的讨论我们可以发现，虽然说我们可以通过为 LLM 提供精细设计的 prompt 以及工具来构造 agent，但要想让 agent 达到良好的效果，需要非常多的工程优化。并且工程优化并没有 silver bullet 式的解法，需要根据工具的实际应用场景来精准地设计。
