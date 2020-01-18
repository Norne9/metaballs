import pygame as pg
import numpy as np
import numba as nb
import os

WIDTH = 800
HEIGHT = 600
CORES = os.cpu_count()
DEFAULT_N_BALLS = 6

@nb.njit
def update_balls(balls: np.ndarray, dt: float):
    # balls -> [[x, y, r, g, b, radius, vx, vy]]
    for b in range(balls.shape[0]):
        radius = balls[b, 5]
        # move
        balls[b, 0:2] += balls[b, 6:8] * dt * 80.0

        # bounce x
        if balls[b, 0] < radius:
            balls[b, 6] = np.abs(balls[b, 6])
        elif balls[b, 0] > WIDTH - radius:
            balls[b, 6] = -np.abs(balls[b, 6])

        # bounce y
        if balls[b, 1] < radius:
            balls[b, 7] = np.abs(balls[b, 7])
        elif balls[b, 1] > HEIGHT - radius:
            balls[b, 7] = -np.abs(balls[b, 7])

        # bounce from others
        for b2 in range(balls.shape[0]):
            if b2 == b:
                continue
            delta = balls[b, 0:2] - balls[b2, 0:2]
            dist2 = delta[0] * delta[0] + delta[1] * delta[1]
            rad2 = balls[b, 5] + balls[b2, 5]
            rad2 *= rad2

            if dist2 < rad2:
                balls[b, 6:8] = delta / np.max(np.abs(delta))


@nb.jit(nopython=True, parallel=True)
def draw_balls(screen: np.ndarray, balls: np.ndarray):
    w, h = screen.shape[0], screen.shape[1]
    b_count = balls.shape[0]

    # to use all cores
    for start in nb.prange(CORES):
        # for each pixel on screen
        for x in range(start, w, CORES):
            for y in range(h):
                screen[x, y].fill(0)  # clear pixel
                # for each ball
                for b in range(b_count):
                    # get ball data from array
                    bx, by = balls[b, 0], balls[b, 1]
                    radius = balls[b, 5]
                    rgb = balls[b, 2:5]

                    # calculate value
                    dx, dy = x - bx, y - by
                    light = radius * radius / (dx * dx + dy * dy)

                    # multiply value by ball color
                    for c in range(3):
                        screen[x, y, c] += rgb[c] * light * 255.0

                # if color > max => normalize color
                max_color = screen[x, y].max()
                if max_color > 255:
                    screen[x, y] = screen[x, y] * 255 // max_color
                else:  # else => color = color * color / 2
                    screen[x, y] *= screen[x, y]
                    screen[x, y] //= 500

def create_balls(n_balls):
    # make random balls
    balls = np.empty((n_balls, 8), dtype=np.float32)
    for i in range(balls.shape[0]):
        # generate ball
        radius = np.random.randint(5, 15) * 5
        x, y = np.random.randint(radius, WIDTH - radius), np.random.randint(radius, HEIGHT - radius)
        color = np.random.rand(3)
        color[i % 3] = 1
        vel = np.random.rand(2)
        vel = vel / vel.max()
        # put ball to array
        balls[i, 0], balls[i, 1] = x, y
        balls[i, 2:5] = color
        balls[i, 5] = radius
        balls[i, 6:8] = vel
    return balls

def run():
    # set seed for repeatability
    np.random.seed(2)
    n_balls = DEFAULT_N_BALLS
    # init pygame
    pg.init()
    pg.font.init()

    # create window
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    # set window title
    pg.display.set_caption(f"Metaballs {WIDTH}x{HEIGHT}")
    # load system font
    font = pg.font.SysFont("Arial", 16, bold=True)

    # necessary variables
    done, fps = False, 0
    clock = pg.time.Clock()

    # show loading
    text = font.render("LOADING...", False, (255, 255, 255))
    screen.blit(text, (16, 16))
    pg.display.flip()

    balls = create_balls(n_balls)
    # numpy array for screen
    screen_arr = np.zeros((WIDTH, HEIGHT, 3), dtype=np.int32)

    while not done:
        dt = clock.tick()  # get elapsed milliseconds
        fps = fps * 0.97 + 1000.0 / max(dt, 1) * 0.03
        dt /= 1000.0

        for event in pg.event.get():  # process events
            if event.type == pg.QUIT:  # clicked close
                done = True  # exit loop
            elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                balls = create_balls(n_balls)
            elif event.type == pg.KEYDOWN and event.key == pg.K_UP:
                n_balls += 1
                balls = create_balls(n_balls)
            elif event.type == pg.KEYDOWN and event.key == pg.K_DOWN:
                n_balls -= 1
                balls = create_balls(n_balls)
        # numpy
        update_balls(balls, dt)  # move balls
        draw_balls(screen_arr, balls)  # draw balls in numpy array
        pg.surfarray.blit_array(screen, screen_arr)  # draw array on screen

        # show fps
        text = font.render(f"FPS: {fps:4.0f}", False, (255, 255, 255))
        screen.blit(text, (16, 16))
        text = font.render(f"PRESS SPACE TO RELOAD. UP TO INCREASE BALL AMOUNT. DOWN TO REDUCE BALL AMOUNT", False, (255, 255, 255))
        screen.blit(text, (16, HEIGHT-16))
        pg.display.flip()

    pg.quit()


if __name__ == "__main__":
    run()
