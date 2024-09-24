### 用法

~~~python
from typing import NamedTuple
class Employee(NamedTuple):
    name: str
    id: int

~~~
相当于 
~~~
Employee = collections.namedtuple('Employee', ['name', 'id'])
~~~

要给一个字段一个默认值，你可以在类体中给它赋值：
~~~
class Employee(NamedTuple):
    name: str
    id: int = 3

employee = Employee('Guido')
assert employee.id == 3
~~~
