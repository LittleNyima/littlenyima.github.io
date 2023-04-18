---
title: 技术相关｜对 Python built-in pprint 进行拓展
date: 2023-03-10 16:48:09
cover: false
categories:
 - Techniques
tags:
 - Python
---

把上一个阶段的工作收尾之后，终于有时间实现一些平时想实现但比较麻烦就一直懒得弄的功能。事情的起因是这样的：刚刚我正在研究 [Human3.6M 数据集](http://vision.imar.ro/human3.6m/description.php)的标注，这个数据集的标注是一个类似 JSON 的格式。并且除了一般的 JSON 文件结构外，其中还有一些字段的值是高维 `numpy` 数组。我试着将其中的两项打印出来，命令行瞬间被一大堆各种各样的数值填满了。尽管 IPython Terminal 的输出有自带的格式化功能，但整个输出的格式还是被巨大的数组打乱了。

实际上我平时经常行地遇到这种问题，一般的解决方法是仿照 `mmcv.parallel.collate` 中的形式写一个递归打印函数，然后把诸如 `np.ndarray`、`torch.Tensor` 以及 `mmcv.DataContainer` 这种高维数据的 `__repr__` 函数映射为一个形式类似 `lambda x: f"{x.shape}, {x.dtype}"` 的函数。

这种方式虽然能在一定程度上解决高维数据打印的问题，但是由于函数逻辑未经仔细考虑，打印的效果依然不是非常好，甚至在一些 corner case 处会出现十分诡异的效果。为了比较彻底地解决这个问题，我决定基于比较成熟的 Python built-in `pprint` 来拓展开发，实现这一功能。

我首先考虑的设计方式是引入类似 `json.JSONEncoder` 的设计方式，也就是在使用 `pprint` 函数时，添加一个额外的参数，将一个 formatter 类传入，对特定类型对象的序列化方式定义在这个类中，类似这样：

```python
class CustomFormatter(DefaultFormatter):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return f"<{obj.__class__.__name__} {obj.dtype} of shape {obj.shape}>"
        return DefaultFormatter.default(self, obj)

pprint.pprint(obj, cls=CustomFormatter)
```

然而观察了一下 `pprint` 的内部实现之后，我发现要改成这种形式还有一点难度。主要是因为在 `pprint` 内部，不同对象的序列化并非依赖于一个 formatter，而是通过维护一个字典，并将不同类型对象的序列化方法注册到这个字典中，在遇到相应对象时直接从字典中找到相应方法进行调用即可，类似这种形式：

```python
_dispatch = {}

def _pprint_list(self, object, stream, indent, allowance, context, level):
    stream.write('[')
    self._format_items(object, stream, indent, allowance + 1,
                       context, level)
    stream.write(']')

_dispatch[list.__repr__] = _pprint_list
```

如此看来，要想将其拓展到更多类别，只需要把 `np.ndarray`、`torch.Tensor` 等类型的 `__repr__` 函数也注册到这一 `_dispatch` 字典中即可。然而此时仍然存在一个问题，这个字典是被所有的 `PrettyPrinter` 对象共享的，因此如果直接在其中添加键值对，会影响 Python 内置方法的行为。我希望能够在维持 built-in 行为不变的前提下添加这一功能，因此在实际进行实现时，我采用了二级 dispatch 字典，在遇到过长对象时，首先在 `_dispatch_override` 中查询，然后再在原有的字典中查询，即可实现重载功能。

为了使这一类型具有较好的可拓展性，我也计划开放一个 `register` 接口，便于用户将更多自定义的类型注册到字典中，以提高更多场景下的可用性。

最后放一个输出的示例（目前还没有适配 `mmcv.parallel.DataContainer`，仅展示 `ndarray` 和 `Tensor` 的效果）：

```python
{'address': {'city': 'New York',
             'postalCode': '10021',
             'state': 'NY',
             'streetAddress': '21 2nd Street'},
 'age': 25,
 'firstName': 'John',
 'lastName': 'Smith',
 'nestedDict': {'nestedList': [<Tensor (shape=torch.Size([233]) dtype=torch.float32)>,
                               <Tensor (shape=torch.Size([114]) dtype=torch.float32)>,
                               <ndarray (shape=(514,) dtype=float64)>],
                'nestedNumpyArray': <ndarray (shape=(123, 456) dtype=float64)>},
 'numpyArray': <ndarray (shape=(3, 4, 5, 6) dtype=float64)>,
 'phoneNumber': [{'number': '212 555-1234', 'type': 'home'},
                 {'number': '646 555-4567', 'type': 'fax'}],
 'sex': 'male',
 'torchTensor': <Tensor (shape=torch.Size([6, 7, 8]) dtype=torch.float32)>}
```

这一模块目前还在开发和测试阶段，预计会在测试稳定后整合进我的 pypi package 中发布，如果想使用预览版，可以参考[这一链接](https://github.com/LittleNyima/python-toolkit/blob/master/pytk/misc/pprint.py)。