import numpy as np
import visdom


def init_visdom_line(x, y, title, xlabel, ylabel, env="default"):
    env = visdom.Visdom(env=env)
    panel = env.line(
        X=np.array([x]),
        Y=np.array([y]),
        opts=dict(title=title, showlegend=True, xlabel=xlabel, ylabel=ylabel)
    )
    return env, panel


def update_lines(env, panel, x, y, update_type='append'):
    env.line(
        X=np.array([x]),
        Y=np.array([y]),
        win=panel,
        update=update_type
    )
