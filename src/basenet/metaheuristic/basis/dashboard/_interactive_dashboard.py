# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import datetime
import numpy as np

import pandas as pd
import hvplot.pandas  # Do not remove.
import panel as pn

CROSS_SECTION_COLOR_1 = '#11B281'
CROSS_SECTION_COLOR_0 = '#00A170'
SIDEBAR_COLOR = CROSS_SECTION_COLOR_1
TITLE_COLOR = '#22C392'
BG_COLOR = TITLE_COLOR


# -----------------------------------------------------------
class Dashboard:
    def __init__(self, constraints, extra_plots: list[tuple] = (), cscope=None, cnt: int = 3, kill: bool = False):
        self.extra_plots = extra_plots
        self.constraints = constraints

        if cscope:
            if cscope.cluster:
                self.cscope_info = {'ip': cscope.cluster.ip,
                                    'c_cpus': cscope.orchestrator.ray_cpus,
                                    'cpus': cscope.orchestrator.cpus,
                                    'seg': cscope.orchestrator.partial}
            else:
                self.cscope_info = {'ip': None,
                                    'c_cpus': None,
                                    'cpus': cscope.orchestrator.cpus,
                                    'seg': 1}
        else:
            self.cscope_info = {'ip': None, 'c_cpus': None, 'cpus': 1, 'seg': 1}

        self.ptotal = None
        self.server = None
        self.kill_after_run = kill
        self.counter = 0
        self.cnt = cnt

        self.dfs = {}
        self.inner_parameters = {}
        self.timestamps = []

        self.performance_ = []
        self.timestamps_ = []
        self.stored_score_ = []
        self.stored_individuals_ = []

    def add(self, score, pop, performance):
        _dtime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        _pop = pop.numpy()
        self.timestamps.append(_dtime)
        self.timestamps_.append(_dtime)
        self.stored_score_.append(score.numpy())
        self.stored_individuals_.append(_pop)
        self.performance_.append(performance)
        self.inner_parameters['i_len'] = len(_pop)

    def refresh_tab(self):
        if self.counter >= self.cnt:
            fv, pp, pf = self.get_df()
            the_args = (fv, pp, pf, self.inner_parameters,
                        self.timestamps, self.cscope_info, self.extra_plots)
            self.render(*the_args)
            self.counter = 0
        else:
            self.counter += 1

    def render(self, fv, pp, pf, inner_parameters, timestamps, cscope_info, extra_plots):
        # Widgets:
        opt_ = []
        based = inner_parameters['i_len']
        basex = 1
        while basex < based:
            opt_.append(basex)
            basex *= 5
        opt_.append(based)
        how_many_indiv = pn.widgets.DiscreteSlider(name='Top individuals',
                                                   options=opt_,
                                                   value=opt_[-1]).servable()
        select_frame = pn.widgets.DiscreteSlider(name='Timestamp for fitness', options=timestamps,
                                                 value=timestamps[-1]).servable()
        select_ts = pn.widgets.DiscreteSlider(name='Timestamp for individuals', options=timestamps,
                                              value=timestamps[0]).servable()

        def get_table(n, ts):
            dataframe = pp[(pp.individual <= n - 1) & (pp.ts == ts)]
            return pn.widgets.DataFrame(dataframe, name='Best individuals', width=765, height=355)

        def get_curve(ts):
            dataframe = pp[pp.ts == ts]
            return dataframe.hvplot.line(x='individual', y='fitness', grid=True, width=765, height=355, title=ts)

        def get_bar(ts):
            idx = timestamps.index(ts)
            dataframe = fv[fv.ts.isin(timestamps[idx:])]
            return dataframe.hvplot.bar(x='ts', value_label='Fitness score', width=765, height=355, grid=True)

        def get_plot(ts):
            idx = timestamps.index(ts)
            dataframe = pf[pf.ts.isin(timestamps[idx:])]
            return dataframe.hvplot.bar(x='ts', stacked=True, width=325, height=222, value_label='Elapsed Time (s)',
                                        legend='top', grid=True)

        def get_fitting_epoch(ts):
            idx = timestamps.index(ts)
            dataframe = pp[pp.ts.isin(timestamps[idx:]) & (pp.individual == 0)]
            return dataframe.hvplot.line(x='ts', y='fitness', value_label='Best Fitness score', grid=True, stacked=True,
                                         width=765, height=355, title=ts, legend=False, color=['red'])

        bound_table = pn.bind(get_table, n=how_many_indiv, ts=select_frame)
        fitness_curve = pn.bind(get_curve, ts=select_frame)
        fitness_bar = pn.bind(get_bar, ts=select_frame)
        performance_plot = pn.bind(get_plot, ts=select_ts)
        fitting_plot = pn.bind(get_fitting_epoch, ts=select_ts)

        ptit = pn.Row(
            pn.pane.Markdown('<center>\n# BaseNetHeuristic Dashboard - A simple way to monitor your'
                             ' MetaHeuristic.\n</center>', width=1800),
            background=TITLE_COLOR
        )
        prow = pn.Row(
            pn.Column(pn.pane.Markdown('# Dashboard visualization panels'),
                      pn.pane.Markdown('## Best individuals'),
                      how_many_indiv,
                      pn.pane.Markdown('## Epoch selection for Fitness'),
                      select_frame,
                      pn.pane.Markdown('## Epoch selection for Performance'),
                      select_ts,
                      pn.pane.Markdown(f'## ComputationalScope information\n'
                                       f'\n*\t**Cluster IP**: \t{cscope_info["ip"]}\n'
                                       f'*\t**Cluster cores**: \t{cscope_info["c_cpus"]}\n'
                                       f'*\t**Own cores**: \t{cscope_info["cpus"]}\n'
                                       f'*\t**Segmentation**: \t{round(cscope_info["seg"], 3)}\n\n\n'
                                       f'### Last epoch: {len(timestamps)} at {timestamps[-1]}'),
                      performance_plot,
                      background=SIDEBAR_COLOR),
            pn.Column(pn.pane.Markdown('<center>\n## Fitness score metrics in the selected TimeStamp\n</center>',
                                       width=765),
                      fitness_bar,
                      pn.pane.Markdown('<center>\n## Best individual over the epochs\n</center>',
                                       width=765),
                      fitting_plot,
                      background=CROSS_SECTION_COLOR_0),
            pn.Column(pn.pane.Markdown('<center>\n## Fitness score for each individual in the selected TimeStamp'
                                       '\n</center>', width=765),
                      fitness_curve,
                      pn.pane.Markdown('<center>\n## Best "n" individuals in the selected TimeStamp\n</center>',
                                       width=765),
                      bound_table,
                      background=CROSS_SECTION_COLOR_1)
        )

        # Extra plots:
        rowings = []
        for user_plot, name in extra_plots:
            widget_individuals = pn.widgets.IntSlider(name='Individual', start=0, end=inner_parameters['i_len'],
                                                      step=1, format='0o').servable()
            widget_epoch = pn.widgets.DiscreteSlider(name='Epoch', options=timestamps,
                                                     value=timestamps[-1]).servable()

            def _user_plot(indi, epoch):
                return user_plot(pp, indi, epoch).opts(width=1550, height=400)

            _user_plot_ = pn.bind(_user_plot, indi=widget_individuals, epoch=widget_epoch)
            rowings.append(pn.Column(
                pn.Row(
                    pn.pane.Markdown(f'<center>\n## User plot\t"{name}"\tin {len(timestamps)} epoch.\n</center>',
                                     width=1870),
                    background=CROSS_SECTION_COLOR_0
                ),
                pn.Row(
                    pn.Column(pn.pane.Markdown('# Sidebar selectors'), widget_individuals, widget_epoch),
                    pn.Column(_user_plot_),
                    background=CROSS_SECTION_COLOR_1
                )
            ))

        if self.ptotal is None:
            self.ptotal = pn.Column(
                pn.Row(ptit),
                pn.Row(prow),
                *rowings,
                background=BG_COLOR
            )
        else:
            self.ptotal.objects = [pn.Column(
                pn.Row(ptit),
                pn.Row(prow),
                *rowings,
                background=BG_COLOR
            )]

        if self.server is None:
            self.server = self.ptotal.show(threaded=True, port=8123, title='BaseNetHeuristic', verbose=False)

    def get_df(self, save=False):
        # Fitness value:
        #
        #       max     avg     min     ts
        #
        #   0   0.9     0.7     0.2     ts0
        #   1   0.99    0.72    0.3     ts1
        #   ... ...     ...     ...     ...
        #   n   1.00    1.00    1.00    tsn

        fitness_value = pd.DataFrame(data=[[max(score) for score in self.stored_score_],
                                           [np.mean(score) for score in self.stored_score_],
                                           [min(score) for score in self.stored_score_],
                                           self.timestamps_],
                                     index=['max', 'avg', 'min', 'ts']).T

        # Population timestamps:
        #
        #       ts      indiv   param0  param1  param2  param3  ... paramM  fitness
        #
        #   0   ts0     0       0.8     0.8     0.9     0.2     ... 0.3     0.45
        #   1   ts0     1       0.6     0.8     0.9     0.2     ... 0.7     0.34
        #   2   ts0     2       0.7     0.6     0.6     0.6     ... 0.1     0.33
        #   ... ...     ...     ...     ...     ...     ...     ... ...
        #   n   tsn     100     0.6     0.8     0.9     0.2     ... 0.7     0.01

        ts = list()
        parameters = dict()
        individual = list()
        fitness = list()
        for numpar, _ in enumerate(self.constraints.parameters):
            parameters[f'param{numpar}'] = []

        for epoch, pop in enumerate(self.stored_individuals_):
            for nind, indi in enumerate(pop):
                for num_p, parameter in enumerate(indi):
                    parameters[f'param{num_p}'].append(parameter)
                ts.append(self.timestamps_[epoch])
                individual.append(nind)
                fitness.append(self.stored_score_[epoch][nind])
        _population = {'ts': ts, 'individual': individual, 'fitness': fitness}
        _population.update(parameters)
        population_ts = pd.DataFrame(_population)

        # Performance dataframe:
        #
        #         fitness     crossover   selection     ts
        #
        #   0     12.0        0.1         0.01          ts0
        #   1     11.3        0.3         0.02          ts1
        #   2     10.8        0.2         0.01          ts2
        #   ...   ...         ...         ...           ...
        #   n     7.89        0.1         0.01          ts3

        perf = self.performance_
        for enumerated, dicted in enumerate(perf):
            dicted.update({'ts': self.timestamps_[enumerated]})
        performance_d = pd.DataFrame(data=perf)

        #   Saving and storing dataframes.
        #

        if not self.dfs:
            self.dfs['fv'] = fitness_value
            self.dfs['pp'] = population_ts
            self.dfs['pf'] = performance_d
        else:
            self.dfs['fv'] = pd.concat([self.dfs['fv'], fitness_value])
            self.dfs['pp'] = pd.concat([self.dfs['pp'], population_ts])
            self.dfs['pf'] = pd.concat([self.dfs['pf'], performance_d])

        if save:
            self.dfs['fv'].to_csv('fitness_value.csv')
            self.dfs['pp'].to_csv('population_ts.csv')
            self.dfs['pf'].to_csv('performance_d.csv')

        self.performance_ = []
        self.stored_score_ = []
        self.stored_individuals_ = []
        self.timestamps_ = []
        return self.dfs['fv'], self.dfs['pp'], self.dfs['pf']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.server is not None and self.kill_after_run:
                self.server.stop()
        except Exception as ex:
            logging.warning(ex)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
