# Logging HOWTO

## Basic Logging Tutorial

logging是追踪当一些软件运行时发生的一些事件的方法。软件开发工程师为他们的代码添加logging调用来表明特定的事件发生过。一个事件通过一个可以选择包含变量信息（例如每一个事件发生的时间可能不同）描述性信息来描述。事件也有开发者归因于事件的重要程度；这个重要性也可以叫做*严重性等级*

## 合适使用logging

logging为简单的logging用途提供了一系列方便的函数。这里有`debug()`, `info()`, `warning()`, `error()`和`critical()`。为了决定何时使用logging，见下表，它说明了对应的常见任务的最好工具。

|你想要进行的任务|对应任务的最好工具|
|--------------|--------------|
|展示命令行脚本或者程序的普通用法的控制台输出|`print()`|
|报告程序正常操作过程中出现的事件（例如状态监控或者错误调查）|`logging.info()`（对于处于诊断目的的非常细节的输出则使用`logging.debug()`）|
|对于一个特定的运行时间提出警告|库代码中的`warning.warn()`，如果问题是可避免的同时客户端应用应该修改来消除这个警告\\`logging.warning()`，如果客户端应用无法做任何事情，但是这个事件应该被注意|
|对于特定的运行时间报告一个错误|产生一个异常|
|在不产生异常的情况下报告一个错误（例如一个长时间服务器进程的错误处理）|`logging.error(), logging.exception(), logging.critical()`，对于特定错误和应用选择合适的方法|

logging函数根据它们追踪的事件的严重性等级被命名。标准的等级和它们的适用性描述如下（严重性等级递增）：

|等级|何时使用|
|----------|-------------|
|`DEBUG`|细节信息，通常只有在诊断问题是对它感兴趣|
|`INFO`|确认事情确实如期待中进行|
|`WARNING`|表明某些不期待的事情发生了，或者表示未来的一些问题（例如磁盘空间低）。软件仍然正常运行|
|`ERROR`|由于更加严重的问题，软件无法运行一些功能|
|`CRITICAL`|严重的错误，表明程序可能不能再运行|

默认的等级是`WARNING`，这表示只有在这个等级或者这个等级以上的事件会被追踪，除非logging包被设置做其他事情

被追踪的事件可以通过不同的方式处理。将追踪的事件打印到控制台是处理它们最简单的方式。另一个常见的方式是将它们写入磁盘文件

## 一个简单的例子

一个非常简单的例子是：

```python
import logging
logging.warning('Watch out!') # 将会打印信息到控制台
logging.info('I told you so) # 不会打印任何东西
```

如果你运行上面这段程序，你将会看到：

```
WARNING:root:Watch out!
```

被输出到控制台。因为默认的等级是`WARNING`，所以`INFO`信息并没有出现。输出的信息包括等级和有logging调用提供的事件描述，也就是'Watch out!'。现在不必担心'root'的部分：我们稍后将会解释。如果你需要，实际的输出可以被非常灵活地格式化；格式的选项也将在之后被解释

## 向文件中logging

将事件的logging记录到文件中是一个常见的情形，这样我们就可以之后查看它。确保在一个新打开的Python解释器尝试一下代码段，不要从上面描述的单元继续：

```python
import logging
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
```

