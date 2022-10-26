# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from basenet import BaseNetCompiler
# -----------------------------------------------------------


def test():
    io_shape = ((12, 12, 1), 12)
    compile_options = {'loss': 'mean_squared_error', 'optimizer': 'adam', 'metrics': 'loss'}
    devs = BaseNetCompiler.show_devs()
    comp = BaseNetCompiler(io_shape=io_shape,
                           compile_options=compile_options,
                           devices=devs)
    comp.add({'Dense': ((128,), {})})
    comp.add({'Dense': ((64,), {})})
    comp.add({'Dense': ((32,), {})})
    comp.compile()
    assert comp.is_compiled


def test2():
    comp = BaseNetCompiler.build_from_yaml(path='../src/basenet/include/config/compilers/base_compiler.yaml')
    comp.compile()
    assert comp.is_compiled


def test3():
    comp = BaseNetCompiler.build_from_yaml()
    comp.compile()
    assert comp.is_compiled


def test4():
    model = BaseNetCompiler.build_from_yaml().compile()
    model.print()
    assert model.is_compiled


if __name__ == '__main__':
    test()
    test2()
    test3()
    test4()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
