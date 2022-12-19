# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    from basenet import BaseNetDatabase, BaseNetCompiler

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

    my_first_compiler = BaseNetCompiler(
        io_shape=((8,), 8),
        compile_options={'loss': 'mean_squared_error', 'optimizer': 'adam'},
        devices=my_devs,
        name='my_first_model'
    )
    for layer in layers:
        my_first_compiler.add(layer)

    my_first_model = my_first_compiler.compile()
    my_first_model.add_database(db_path='mydb.db')

    mydb = BaseNetDatabase.load('./mydb.db')
    print(mydb)

    #    > BaseNetDatabase with 32000 instances.

    yaml_path = 'my_model.yaml'

    my_first_model = BaseNetCompiler.build_from_yaml(yaml_path).compile()
    my_first_model.add_database(mydb)

    from basenet import BaseNetDatabase, BaseNetCompiler

    mydb = BaseNetDatabase.load('./mydb.db')
    my_first_model = BaseNetCompiler.build_from_yaml('./my_model.yaml', verbose=True).compile()
    my_first_model.add_database(mydb)

    # Select database with index 0.
    my_first_model.fit(0, epochs=6, tensorboard=False)

    #    >   Tensorflow fitting info vomiting.

    # Print the model.
    my_first_model.print('./')
    from basenet import BaseNetDatabase, BaseNetCompiler

    mydb = BaseNetDatabase.load('./mydb.db')
    my_first_model = BaseNetCompiler.build_from_yaml('./my_model.yaml', verbose=True).compile()
    my_first_model.add_database(mydb)

    # Select database with index 0.
    my_results = my_first_model.fit(epochs=6, tensorboard=False, avoid_lock=True)

    while my_results.is_training:
        # do_my_main_activity(update_gui, collect_data, run_server, or_whatever)
        current_loss_curve = my_results.get()
    my_first_model.recover()

    # keep_doing_my_main_activity(update_gui, collect_data, run_server, or_whatever)
    from basenet import BaseNetDatabase, BaseNetCompiler
    mydb = BaseNetDatabase.load('./mydb.db')
    my_first_model = BaseNetCompiler.build_from_yaml('./my_model.yaml', verbose=True).compile()
    my_second_model = BaseNetCompiler.build_from_yaml('./my_model_2.yaml', verbose=True).compile()
    my_first_model.add_database(mydb)

    my_first_model(my_second_model, parallel=True, name='merged_model')
    my_first_model.print('./')
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
