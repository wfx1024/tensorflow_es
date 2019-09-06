# -*- coding:utf-8 -*-

"""
常量定义
"""


class Const():
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, key, value):
        print()
        if key in self.__dict__.keys():
            # 存在性验证
            raise self.ConstError("常量不能修改: '%s'" % key)

        if not key.isupper():
            # 语法规范验证
            raise self.ConstCaseError("Const variable must be combined with upper letters:'%s'" % key)

        self.__dict__[key] = value
