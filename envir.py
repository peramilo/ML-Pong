import numpy as np
import pygame


class Game:
    def __init__(self):
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.screenWidth = 800
        self.screenHeight = 600
        self.padWidth = 100
        self.padHeight = 15
        self.ballRad = 12
        self.padPosX = 0
        self.padPosY = self.screenHeight - 10
        self.padPosX = int(self.screenWidth / 2 - self.padWidth / 2)
        self.ballPosX = int(self.screenWidth / 2 - self.ballRad / 2)
        self.ballPosY = int(self.screenHeight / 6)
        self.padVel = 0
        self.ballVelX = int(np.random.choice((-5, -4, -3, 3, 4, 5), 1))
        self.ballVelY = 4
        self.point = 0
        self.reward = 0
        self.done = False
        # For displaying attempts we need to uncomment screen and allow draw_screen in main_loop
        # Using Clock we can limit the fps and make it easier to see model's actions
        # self.fps = pygame.time.Clock()
        # self.screen = pygame.display.set_mode((self.screenWidth, self.screenHeight))

    def draw_screen(self):
        self.screen.fill(self.BLACK)
        self.draw_ball()
        self.draw_pad()
        pygame.display.flip()

    def draw_ball(self):
        pygame.draw.circle(self.screen, self.WHITE, (self.ballPosX, self.ballPosY), self.ballRad, 0)

    def draw_pad(self):
        pygame.draw.rect(self.screen, self.WHITE, (self.padPosX, self.padPosY, self.padWidth, self.padHeight))

    def move_ball(self):
        self.ballPosX += self.ballVelX
        self.ballPosY += self.ballVelY

    def move_pad(self):
        if 0 <= self.padPosX <= self.screenWidth - self.padWidth:
            self.padPosX += self.padVel
        else:
            if self.padPosX <= 0:
                self.padPosX = 1
            else:
                self.padPosX = self.screenWidth - self.padWidth - 1
        self.padVel = 0

    def action(self, x):
        if x == 1:
            self.padVel += 4
        else:
            if x == 0:
                self.padVel -= 4

    def coll(self):
        if self.ballPosY - self.ballRad <= 0:
            self.ballPosY = self.ballRad
            self.ballVelY *= -1
        if self.ballPosX + self.ballRad >= self.screenWidth or self.ballPosX - self.ballRad <= 0:
            self.ballVelX *= -1
        if self.ballPosY + self.ballRad >= self.screenHeight:
            if self.ballPosX + self.ballRad >= self.padPosX and self.ballPosX - self.ballRad <= self.padPosX + self.padWidth:
                self.pad_coll()
            else:
                self.reward = -1000
                self.done = True

        if self.ballPosX + self.ballRad >= self.padPosX and self.ballPosX - self.ballRad <= self.padPosX + self.padWidth:
            if self.ballPosY + self.ballRad >= self.padPosY:
                self.pad_coll()

        # Gives reward of 1 for each frame where we are moving towards the ball
        if abs(self.padPosX + self.padWidth / 2 + self.padVel - self.ballPosX) <= abs(
                self.padPosX + self.padWidth / 2 - self.ballPosX):
            if self.reward == 0:
                self.reward = 1
                
    def pad_coll(self):
        self.ballVelY *= -1
        self.point += 1
        self.reward = 200
        self.ballPosY = self.padPosY - self.ballRad

    def state(self):
        x = (self.ballPosX / self.screenWidth, self.ballPosY / self.screenHeight,
             (self.padPosX + self.padWidth / 2) / self.screenWidth)
        return x

    def reset(self):
        self.__init__()
        self.ballVelX = int(np.random.choice((-4, -3, 3, 4), 1))
        self.ballVelY = 4
        self.done = False
        self.point = 0

    def get_reward(self):
        x = self.reward
        self.reward = 0
        return x

    def is_done(self):
        return self.done

    def main_loop(self):
        self.coll()
        self.move_ball()
        # Uncomment for limiting fps
        # self.fps.tick(30)
        self.move_pad()
        # Uncomment for displaying attempts
        # self.draw_screen()

