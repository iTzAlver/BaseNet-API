# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import os
__config_path__ = os.path.abspath(f'{__file__.replace(f"__special__.py", "")}/../../config/config.json')
__base_compiler__ = os.path.abspath(f'{__file__.replace(f"__special__.py", "")}/../../config/'
                                    f'compilers/base_compiler.yaml')
__temp_path__ = os.path.abspath(f'{__file__.replace(f"__special__.py", "")}/../../temp/')
__keras_checkpoint__ = f'{__temp_path__}/checkpoints''/model.{epoch:02d}-{val_loss:.2f}.h5'
__tensorboard_logs__ = f'{__temp_path__}/logs''/model.{epoch:02d}-{val_loss:.2f}.h5'
__print_model_path__ = f'{__temp_path__}/render/'
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
