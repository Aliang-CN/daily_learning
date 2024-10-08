### 1. 是类还是函数
几乎所有的教程都告诉你int()是python的内置函数，然而当你看到int的定义，发现它竟然是类
~~~
class int(object):
    pass
~~~
不只是int(),还有float()， bool()， str()，很多你以为是函数但实际上却是类，但是呢，当你使用它们时完全察觉不出它们有什么不同，本文不是要和你讨论类和函数，而是要讨论学习callable

### 2. 什么是callable
一个可callable的对象是指可以被调用执行的对象，并且可以传入参数， 用另一个简单的描述方式，只要可以在一个对象的后面使用小括号来执行代码，那么这个对象就是callable对象，下面列举callable对象的种类

- 函数
- 类
- 类里的函数

实现了__call__方法的实例对象
#### 2.1 函数
~~~
def test():
    print('ok')

print(callable(test))   # True
test()  # ok
~~~
函数是python里的一等公民，函数是可调用对象，使用callable函数可以证明这一点

#### 2.2 类
~~~
class Stu(object):
    def __init__(self, name):
        self.name = name

print(callable(Stu))     # True
print(Stu('小明').name)   # 小明
~~~
在其他编程语言里，类与函数可以说是两个完全不搭的东西，但在python里，都是可调用对象

2.3 类里的方法
类里的方法也是用def定义的，本质上也是函数
~~~
from inspect import isfunction, ismethod


class Stu(object):
    def __init__(self, name):
        self.name = name

    def run(self):
        print('{name} is running'.format(name=self.name))

print(isfunction(Stu.run))     # True
stu = Stu("小明")
stu.run()        # 小明 is running
~~~
使用isfunction函数可以判断一个对象是否是函数，run方法也是可调用对象

2.4 实现了__call__方法的实例对象
~~~
class Stu(object):

    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        self.run()

    def run(self):
        print('{name} is running'.format(name=self.name))

stu = Stu('小明')
print(callable(stu))    # True
stu()                   # 小明 is running
~~~
当你执行stu()时，与调用一个函数有着完全一致的体验，如果不告诉你stu是一个类的实例对象，你还以为stu就是一个函数。