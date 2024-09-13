### 前言
Field 可用于提供有关字段和验证的额外信息，如设置必填项和可选，设置最大值和最小值，字符串长度等限制

### Field模块
关于 Field 字段参数说明
- Field(None) 是可选字段，不传的时候值默认为None
- Field(…) 是设置必填项字段
- title 自定义标题，如果没有默认就是字段属性的值
- description 定义字段描述内容

~~~python
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str
    description: str = Field(None,
                             title="The description of the item",
                             max_length=10)
    price: float = Field(...,
                         gt=0,
                         description="The price must be greater than zero")
    tax: float = None

a = Item(name="yo yo",
         price=22.0,
         tax=0.9)
print(a.dict())  # {'name': 'yo yo', 'description': None, 'price': 22.0, 'tax': 0.9}
~~~

##### schema_json
title 和 description 在 schema_json 输出的时候可以看到

~~~python
print(Item.schema_json(indent=2))

"""
{
  "title": "Item",
  "type": "object",
  "properties": {
    "name": {
      "title": "Name",
      "type": "string"
    },
    "description": {
      "title": "The description of the item",
      "maxLength": 10,
      "type": "string"
    },
    "price": {
      "title": "Price",
      "description": "The price must be greater than zero",
      "exclusiveMinimum": 0,
      "type": "number"
    },
    "tax": {
      "title": "Tax",
      "type": "number"
    }
  },
  "required": [
    "name",
    "price"
  ]
}
"""
~~~

