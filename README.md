# BaseNet: A simpler way to build AI models.

<p align="center">
    <img src="https://raw.githubusercontent.com/iTzAlver/basenet_api/main/doc/multimedia/basenet_logo.png">
</p>

<p align="center">
    <a href="https://github.com/iTzAlver/basenet_api/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/iTzAlver/basenet_api?color=purple&style=plastic" /></a>
    <a href="https://github.com/iTzAlver/basenet_api/tree/main/test">
        <img src="https://img.shields.io/badge/tests-passed-green?color=green&style=plastic" /></a>
    <a href="https://github.com/iTzAlver/basenet_api/blob/main/requirements.txt">
        <img src="https://img.shields.io/badge/requirements-pypi-red?color=red&style=plastic" /></a>
    <a href="https://htmlpreview.github.io/?https://github.com/iTzAlver/basenet_api/blob/main/doc/basenet.html">
        <img src="https://img.shields.io/badge/doc-available-green?color=yellow&style=plastic" /></a>
    <a href="https://github.com/iTzAlver/basenet_api.git">
        <img src="https://img.shields.io/badge/release-0.2.0-white?color=white&style=plastic" /></a>
</p>

<p align="center">
    <a href="https://www.tensorflow.org/">
        <img src="https://img.shields.io/badge/dependencies-tensorflow-red?color=orange&style=for-the-badge" /></a>
    <a href="https://keras.io/">
        <img src="https://img.shields.io/badge/dependencies-keras-red?color=red&style=for-the-badge" /></a>
</p>

# Basenet API Package - 0.2.0

This package implements an API over Keras and Tensorflow to build Deep Learning models easily without losing the
framework flexibility. BaseNet API tries to implement almost everything from a few lines of code.

## About ##

    Author: A.Palomo-Alonso (a.palomo@uah.es)
    Universidad de Alcalá.
    Escuela Politécnica Superior.
    Departamento de Teoría De la Señal y Comunicaciones (TDSC).
    ISDEFE Chair of Research.

## Features

* **Feature 1:** Real-time logging.
* **Feature 2:** Database train, validation and test automatic and random segmentation.
* **Feature 3:** Real multiprocessing training process.
* **Feature 4:** Automatic and custom GPU usage.
* **Feature 5:** Easy-to-use classes.
* **Feature 6:** 
* **Feature 7:** 
* **Feature 8:** 
* **Feature x:** API documentation.
* **Feature x:** Python Packaging and PyPi indexing.

## Basic and fast usage

### BaseNetDataset

BaseNetDatabase is an easy-to-use database wrapper for the API. You can build your 
database with the BaseNetDatabase class.

### Example of building a BaseNetDataset.

    from basenet import BaseNetDatabase

    my_data_x, my_data_y = load_my_data()
    print(my_data_y)

    #    > array([[0.], [1.], ...], dtype=float32)

    print(my_data_x)

    #    > array([[255., 0., 255., ..., dtype=float32)

    distribution = {'train': 60, 'test': 5, 'val': 35}
    mydb = BaseNetDatabase(x, y, 
                           distribution=distribution)
    
    print(mydb)

    #    > BaseNetDatabase with 32000 instances.

    mydb.save('./mydb.db')

### BaseNetCompiler

BaseNetCompiler takes the model architecture and builds a BaseNetModel with the given
parameters. You can build your BaseNetCompiler from Python code only or a .yaml file.

### Example of building a BaseNetCompiler from Python code only.

    from basenet import BaseNetDatabase, BaseNetCompiler

    mydb = BaseNetDatabase.load('./mydb.db')
    print(mydb)

    #    > BaseNetDatabase with 32000 instances.

    layers = [
        {'Dense': (255, {})},
        {'Dense': (64, {'activation': 'relu'})},
        {'Dropout': (0.5, {})}
    ]

    my_devs = BaseNetCompiler.show_devs()
    print(my_devs)
    
    #    > {'/device:CPU:0': 'Idle', 
    #       '/device:GPU:0': 'Train'}

    my_first_model = BaseNetCompiler(
        io_shape=((8,), 8), 
        compile_options={'loss': 'mean_squared_error', 'optimizer': 'adam'}, 
        devices=my_devs, 
        layers=layers,
        name='my_first_model'
    ).compile()

    my_fitst_model.add_database(mydb)

You can also use the BaseNetModel.add() method to add layers.

    my_first_compiler = BaseNetCompiler(
        io_shape=((8,), 8), 
        compile_options={'loss': 'mean_squared_error', 'optimizer': 'adam'}, 
        devices=my_devs,
        name='my_first_model'
    )
    for layer in layers:
        my_first_compiler.add(layer)
    
    my_first_model = my_first_compiler.compile()

You can also load the database from the path.

    my_fitst_model.add_database('./mydb.db')

### Example of building a BaseNetCompiler from .yaml file.

Suppose you have a ``.yaml`` file in the ``./my_model.yaml`` location with
the proper format you can load your compiler with the method ``BaseNetCompiler().build_from_yaml(yaml_path)``
and omit the process of loading the parameters into the compiler manually.

    from basenet import BaseNetDatabase, BaseNetCompiler

    mydb = BaseNetDatabase.load('./mydb.db')
    print(mydb)

    #    > BaseNetDatabase with 32000 instances.

    yaml_path = './my_model.yaml'

    my_first_model = BaseNetCompiler().build_from_yaml(yaml_path).compile()
    my_fitst_model.add_database(mydb)

An example of ``.yaml`` to replicate the same model as in the section 
``Building a BaseNetCompiler from Python code only.``, the ``.yaml`` file will be the following:
    

    compiler:
      name: "my_first_model"
      input_shape:
        - 8
      output_shape: 8
    
      compile_options:
        loss: "mean_squared_error"
        optimizer: "adam"
    
      devices:
        - cpu:
            name: "/device:CPU:0"
            state: "Idle"
        - gpu:
            name: "/device:GPU:0"
            state: "Train"
    
      layers:
        - layer:
            name: "Dense"
            shape:
              - 128
            options:
    
        - layer:
            name: "Dense"
            shape:
              - 128
            options:
              - option:
                  name: "activation"
                  value: "relu"
    
        - layer:
            name: "Dropout"
            shape:
              - 0.5
            options:

If you want to learn more about building a model from a ```.yaml``` file, please, check the API 
[documentation](https://htmlpreview.github.io/?https://github.com/iTzAlver/basenet_api/blob/main/doc/basenet.html).

### Example of usage of the BaseNetModel.



## What's new?

### < 0.1.0
1. BaseNetModel included.
2. BaseNetDatabase included.
3. BaseNetCompiler included.
4. Inheritance from CorNetAPI project.
5. Multi-processing fitting.
6. Tensorboard launching.

### 0.2.0
1. BaseNetResults included (working).
2. Now the model is callable.
3. Switched print to logging.
4. Project documentation.


### Cite as

    @misc{cornetapi,
      title={CorNet: Correlation clustering solving methods based on Deep Learning Models},
      author={A.Palomo-Alonso, S.Jiménez-Fernández, S.Salcedo-Sanz},
      booktitle={PhD in Telecommunication Engeneering},
      year={2022}
    }
