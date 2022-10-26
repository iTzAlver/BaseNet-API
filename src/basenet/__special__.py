# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import os
__version__ = '1.0'
__config_path__ = os.path.abspath(f'{__file__.replace(f"__special__.py", "")}/.config/config.json')
__base_compiler__ = os.path.abspath(f'{__file__.replace(f"__special__.py", "")}/.config/'
                                    f'compilers/base_compiler.yaml')
__temp_path__ = os.path.abspath(f'{__file__.replace(f"__special__.py", "")}/.temp/')
__keras_checkpoint__ = f'{__temp_path__}/checkpoints''/model.'
__tensorboard_logs__ = f'{__temp_path__}/logs'
__print_model_path__ = f'{__temp_path__}/render/'
__bypass_path__ = f'{__temp_path__}/bypass/bypass.h5'
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
