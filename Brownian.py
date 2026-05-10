import numpy as np
from matplotlib import pyplot as pl


def particle_simulation(n_particles, n_steps):
    loc = 0
    std = 1
    rng = np.random.default_rng()
    sim = rng.normal(loc, std, size=(n_particles, n_steps))
    trajectory = np.cumsum(sim, axis=1)
    return trajectory


def compute_msd(trajectory):
    msd = np.mean(trajectory**2, axis=0)
    return msd


def plot_result(time, msd):
    pl.plot(time, msd, 'b', label="Симуляция")
    pl.plot(time, time, 'r-', label="теория")
    pl.title("Сравнение среднего квадрата отклонения частиц (симуляция vs теория)")
    pl.xlabel("Время")
    pl.ylabel("Средний квадрат отклонения")
    pl.grid(True)
    pl.legend()
    pl.xlim(0, max(time))
    pl.ylim(0, max(msd) * 1.05)


def main():
    try:
        n_particles = int(input("Количество частиц: "))
        n_steps = int(input("Количество шагов: "))
        if n_particles <= 0 or n_steps <= 0:
            raise ValueError
    except ValueError:
        print('Введите целые натуральные числа')
        return
    trajectories = particle_simulation(n_particles, n_steps)
    msd = compute_msd(trajectories)
    time = np.arange(len(msd))
    plot_result(time, msd)
    pl.show()
    expected = n_steps
    tolerance = 0.05 * n_steps
    if abs(msd[-1] - expected) < tolerance:
        print(
            f"Закон Эйнштейна подтвержден. msd({expected}) = {msd[-1]:.1f}, ожидалось {expected} +- {tolerance:.0f}")
    else:
        print(
            "Отклонение слишком велико. msd({expected}) = {msd[-1]:.1f}, ожидалось {expected} +- {tolerance:.0f}")


if __name__ == "__main__":
    main()
