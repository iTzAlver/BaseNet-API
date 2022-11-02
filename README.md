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
    <a href="https://github.com/iTzAlver/BaseNet-API/releases/tag/1.5.0-release">
        <img src="https://img.shields.io/badge/release-1.5.0-white?color=white&style=plastic" /></a>
</p>

<p align="center">
    <a href="https://www.tensorflow.org/">
        <img src="https://img.shields.io/badge/dependencies-tensorflow-red?color=orange&style=for-the-badge" /></a>
    <a href="https://keras.io/">
        <img src="https://img.shields.io/badge/dependencies-keras-red?color=red&style=for-the-badge" /></a>
</p>

# Basenet API Package - 1.5.1

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
* **Feature 6:** Model merging.
* **Feature 7:** Multiple model inputs.
* **Feature 8:** API documentation.
* **Feature 9:** Python Packaging and PyPi indexing.
* **Feature 10:** Automatic GPU configuration and assignment.

## Basic and fast usage

### BaseNetDataset

BaseNetDatabase is an easy-to-use database wrapper for the API. You can build your 
database with the BaseNetDatabase class.

### Example of building a BaseNetDataset.

    from basenet_api import BaseNetDatabase

    my_data_x, my_data_y = load_my_data()
    print(my_data_y)

    #    > array([[0.], [1.], ...], dtype=float32)

    print(my_data_x)

    #    > array([[255., 0., 255., ..., dtype=float32)

    distribution = {'train': 60, 'test': 5, 'val': 35}
    mydb = BaseNetDatabase(my_data_x, my_data_y, 
                           distribution=distribution)
    
    print(mydb)

    #    > BaseNetDatabase with 32000 instances.

    mydb.save('./mydb.db')

### BaseNetCompiler

BaseNetCompiler takes the model architecture and builds a BaseNetModel with the given
parameters. You can build your BaseNetCompiler from Python code only or a .yaml file.

### Example of building a BaseNetCompiler from Python code only.

    from basenet_api import BaseNetDatabase, BaseNetCompiler

    mydb = BaseNetDatabase.load('./mydb.db')
    print(mydb)

    #    > BaseNetDatabase with 32000 instances.

    layers = [
        {'Dense': ((255,), {})},
        {'Dense': ((64,), {'activation': 'relu'})},
        {'Dropout': ((0.5,), {})}
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

    my_first_model.add_database(mydb)

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

    from basenet_api import BaseNetDatabase, BaseNetCompiler

    mydb = BaseNetDatabase.load('./mydb.db')
    print(mydb)

    #    > BaseNetDatabase with 32000 instances.

    yaml_path = './my_model.yaml'

    my_first_model = BaseNetCompiler.build_from_yaml(yaml_path).compile()
    my_first_model.add_database(mydb)

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

Once you build and compile a ``BaseNetModel`` with a ```BaseNetCompiler.compile()``` method, you can make use of all the
methods that the BaseNetModel provides:

* ```BaseNetModel.load()```: This method loads a tf.keras.model and the compiler from the given path.
* ```BaseNetModel.save()```: This method saves a tf.keras.model and the compiler into the given path.
* ```BaseNetModel.print()```: This method renders a ``.png`` image of the model into the given path.
* ```BaseNetModel.add_database()```: The ``BaseNetModel`` contains a breech of databases. It is a list with all the loaded
databases previously. This method adds a database from a path or from a ```BaseNetDatabase``` object.
* ```BaseNetModel.predict()```: Performs a prediction given an input.
* ```BaseNetModel.evaluate()```: Evaluates the model with the pointed database test subset.
* ```BaseNetModel.fit()```: Trains the model with the pointed database.
* ```BaseNetModel.call()```: Merges two models into one. It can be used as a function.

#### Printing and fitting a model.

    from basenet_api import BaseNetDatabase, BaseNetCompiler

    mydb = BaseNetDatabase.load('./mydb.db')
    my_first_model = BaseNetCompiler.build_from_yaml('./my_model.yaml').compile()
    my_first_model.add_database(mydb)

    # Select database with index 0.
    my_first_model.fit(0, epochs=6, tensorboard=False)

    #    >   Tensorflow fitting info vomiting.

    # Print the model.
    my_first_model.print('./my_model.png')

<p align="center">
    <img src="https://raw.githubusercontent.com/iTzAlver/basenet_api/main/doc/multimedia/example_model.png">
</p>

#### Fitting a model in other process.

**Important:** Debugging is not working properly when fitting a new process.

Imagine working on a GUI. The training process of your model implemented on your
GUI will block the parent process. The API implements a solution. Just activate
``avoid_lock=True`` in the ``BaseNetModel.fit()`` method and check the results whenever you want.

    from basenet_api import BaseNetDatabase, BaseNetCompiler

    mydb = BaseNetDatabase.load('./mydb.db')
    my_first_model = BaseNetCompiler.build_from_yaml('./my_model.yaml').compile()
    my_first_model.add_database(mydb)

    # Select database with index 0.
    my_results = my_first_model.fit(0, epochs=6, tensorboard=False, avoid_lock=True)

    while my_results.is_training:
        do_my_main_activity(update_gui, collect_data, run_server, or_whatever)
        current_loss_curve = my_results.get()

    # my_first_model.recover() Use it in versions < 1.5.0.

    keep_doing_my_main_activity(update_gui, collect_data, run_server, or_whatever)

```OutDated```:
Note that if you don't make use of the method ``BaseNetModel.recover()`` the model will be empty as
the trained model is bypassed by the child process until the parent process is able to recover the trained model.

```From >= 1.5.0```: The model recovers itself, there is no need (or ways) to recover it manually.

#### Using Tensorboard.

The API also implements Tensorboard automatic opening and initialization. You can see the training process and keras
app in real time while training.

    my_first_model.fit(0, epochs=6, tensorboard=True)

#### Merging two models into one with several inputs.

You can merge two BaseNetModels by calling the object as a function:

    from basenet_api import BaseNetDatabase, BaseNetCompiler
    mydb = BaseNetDatabase.load('./mydb.db')
    my_first_model = BaseNetCompiler.build_from_yaml('./my_model.yaml', verbose=True).compile()
    my_second_model = BaseNetCompiler.build_from_yaml('./my_model_2.yaml', verbose=True).compile()
    my_first_model.add_database(mydb)

    my_first_model(my_second_model, parallel=True, name='merged_model')
    my_first_model.print('./')

It will merge the two models into one single with two outputs if ``parallel=True``, else it will be added at the bottom.

<p align="center">
    <img src="https://raw.githubusercontent.com/iTzAlver/basenet_api/main/doc/multimedia/example_model2.png">
</p>

#### Obtaining training results from the fitting process.

Once you train the model, you can get a ``BaseNetResults`` object with the training results. You can obtain the values from:

    my_results = my_first_model.fit(0, epochs=6)
    losses = my_results.get()
    print(losses)

    #    > {'loss': [1., 0.7, 0.6, 0.5, 0.4, 0.3], 
    #       'val_loss': [1., 0.8, 0.7, 0.6, 0.5, 0.4]}

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


### 1.0.0 - 1.0.3
1. Python packaging
3. 1.0.x: Upload bug solving.

### 1.1.0
1. Functional package.
2. PyPi indexing.

### 1.2.0:
1. Loss results included in the BaseNetResults while multiprocessing.
2. GPU auto set up to avoid TensorFlow memory errors.
3. Method ``BaseNetCompiler.set_up_devices()`` configures the GPUs according to the free RAM to be used in the API.

### 1.3.0
1. Included WindowDiff to the project scope.

### 1.4.0
1. Solved python packaging problems.
2. Included force stop callback in the ```BaseNetModel.fit_stop()``` method.

### 1.5.0
1. BaseNetDatabase now has the attributes ``BaseNetDatabase.size`` and ``BaseNetDatabase.distribution``.
2. Solved forced stopping bugs with multiprocessing in the method ``BaseNetDatabase.fit_stop()``.
3. ```BaseNetModel._threshold()``` private method now takes a set of outputs instead only one. 
This was only for optimization.
4. Solved wrong ```BaseNetModel.recover()```.
5. **Auto recover implemented**, now ```BaseNetModel.recover()``` is a private method: ```BaseNetModel._recover()```.
Now the used does not need to recover it. *The model recovers by itself. -- Hans Niemann 2022.*

### 1.5.1
1. Solved a bug where ``BaseNetDatabase`` modified the incoming list of instances in the database; avoiding checkpoints
for large database generators.

### Cite as

Please, cite this library as:


    @misc{basenetapi,
      title={CorNet: Correlation clustering solving methods based on Deep Learning Models},
      author={A. Palomo-Alonso},
      booktitle={PhD in TIC: Machine Learning and NLP.},
      year={2022}
    }
