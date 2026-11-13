import random

PADDLE_TOP = 460
PADDLE_BOTTOM = 472


def reaches_paddle(speed, crossing):
    """Drop the ball's bottom edge straight down from a random height.
    True if the check sees it reach the paddle."""
    bottom = PADDLE_TOP - 200 - random.uniform(0, speed)
    while bottom < PADDLE_BOTTOM + 100:
        before = bottom
        bottom += speed
        if crossing:
            hit = bottom >= PADDLE_TOP and before <= PADDLE_BOTTOM
        else:
            hit = PADDLE_TOP <= bottom <= PADDLE_BOTTOM
        if hit:
            return True
    return False


random.seed(1)
for speed in [6, 12, 13.5, 15, 16]:
    caught_old = 0
    caught_new = 0
    for drop in range(1000):
        if reaches_paddle(speed, crossing=False):
            caught_old += 1
        if reaches_paddle(speed, crossing=True):
            caught_new += 1
    print("speed", speed, " old:", caught_old, " new:", caught_new)
