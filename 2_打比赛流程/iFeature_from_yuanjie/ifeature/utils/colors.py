# -*- coding: utf-8 -*-
"""
__title__ = 'colors'
__author__ = 'JieYuan'
__mtime__ = '2018/6/7'
"""


class Cprint(object):
    def __init__(self):
        self.foreground_color = {
            'black': 90,
            'red': 91,
            'green': 92,
            'yellow': 93,
            'blue': 94,
            'purple': 95,
            'cyan': 96,
            'white': 97
        }

        self.background_color = {
            'black': 40,
            'red': 41,
            'green': 42,
            'yellow': 43,
            'blue': 44,
            'purple': 45,
            'cyan': 46,
            'white': 47
        }

    def cprint(self, obj='Hello World!!!', foreground='red', background='blue', mode=0):
        """
        :param obj: 可字符化
        :param foreground:
            'black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white'
        :param background:
            'black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white'
        :param mode:
            0（默认值）、1（高亮）、22（非粗体）、4（下划线）、24（非下划线）、 5（闪烁）、25（非闪烁）、7（反显）、27（非反显）
        :return:
        """

        print('\033[%s;%s;%sm%s\033[0m' % (mode, self.foreground_color[foreground], self.background_color[background], obj))


if __name__ == '__main__':
    Cprint().cprint()
