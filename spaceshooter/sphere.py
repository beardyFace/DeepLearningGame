#!/usr/bin/env python
import math
import pygame
from math import sqrt
from learner import DeepLearner

class Sphere(object):
  def __init__(self, x, y, t, max_vel, max_acc, size, vel=0):
    self.limts = {}
    self.limts['size'] = size 
    self.setValues(x, y, t, max_vel, max_acc, vel)
    
  def setValues(self, x, y, t, max_vel, max_acc, vel=0):
    self.limts['max_vel'] = max_vel
    self.limts['max_acc'] = max_acc

    self.pose = {}
    self.pose['x'] = x
    self.pose['y'] = y
    self.pose['t'] = t

    self.pose['vel_x'] = 0
    self.pose['vel_y'] = 0
    self.setVelocities(vel)
    
  def tick(self, display_width, display_height):
    self.move(display_width, display_height)

  def rotateLeft(self):
    self.rotate(-self.limts['max_acc'] * 20)

  def rotateRight(self):
    self.rotate(self.limts['max_acc'] * 20)

  def rotate(self, dt):
    self.pose['t'] += dt
    if self.pose['t'] > 359:
        self.pose['t'] -= 360
    if self.pose['t'] < 0:
        self.pose['t'] += 360

  def accelerate(self):
    self.setVelocities(self.limts['max_acc'])

  def deccelerate(self):
    self.setVelocities(-self.limts['max_acc'])

  def move(self, display_width, display_height):
    self.pose['x'] += self.pose['vel_x']
    self.pose['y'] += self.pose['vel_y']

    if self.pose['x'] >= display_width:
      self.pose['x'] = display_width
      self.pose['vel_x'] = 0
    elif self.pose['x'] < 0:
      self.pose['x'] = 0
      self.pose['vel_x'] = 0

    if self.pose['y'] >= display_height:
      self.pose['y'] = display_height
      self.pose['vel_y'] = 0
    elif self.pose['y'] < 0:
      self.pose['y'] = 0
      self.pose['vel_y'] = 0

  def setVelocities(self, acc):
    theta = self.pose['t']
    self.pose['vel_x'] += acc * math.cos(math.radians(theta))
    self.pose['vel_y'] += acc * math.sin(math.radians(theta))

    if self.pose['vel_x'] > self.limts['max_vel']:
        self.pose['vel_x'] = self.limts['max_vel']
    elif self.pose['vel_x'] < -self.limts['max_vel']:
        self.pose['vel_x'] = -self.limts['max_vel']
    
    if self.pose['vel_y'] > self.limts['max_vel']:
        self.pose['vel_y'] = self.limts['max_vel']
    elif self.pose['vel_y'] < -self.limts['max_vel']:
        self.pose['vel_y'] = -self.limts['max_vel']

  def render(self, gameDisplay):
    pygame.draw.circle(gameDisplay, (0,255,255), (int(self.pose['x']), int(self.pose['y'])), self.limts['size'])
    # pygame.draw.circle(gameDisplay, (0,0,255), (int(self.pose['x']), int(self.pose['y'])), self.limts['size']/2)

  def checkCollision(self,sphere):
    x_diff = self.pose['x'] - sphere.pose['x']
    y_diff = self.pose['y'] - sphere.pose['y']
    hyp = sqrt(x_diff**2 + y_diff**2)
    
    return hyp <= (self.limts['size'] + sphere.limts['size'])


#############################################################
class Ship(Sphere):
  def __init__(self, x, y, t, max_vel, max_acc, vel=0):
    Sphere.__init__(self, x, y, t, max_vel, max_acc, 40, vel)

    self.deep_learner = DeepLearner()

  def fire(self):
    theta = self.pose['t']
    x = self.pose['x'] + 2 * self.limts['size'] * math.cos(math.radians(theta))
    y = self.pose['y'] + 2 * self.limts['size'] * math.sin(math.radians(theta))
    t = self.pose['t']
    max_acc = 0
    ship_vel = sqrt(self.pose['vel_x']**2 + self.pose['vel_y']**2)
    vel = 5 + ship_vel

    max_vel = vel
    bullet = Bullet(x, y, t, max_vel, max_acc, vel)
    return bullet

  def render(self, gameDisplay):
    super(Ship, self).render(gameDisplay)
    theta = self.pose['t']
    x = self.pose['x'] + (self.limts['size'] + 0) * math.cos(math.radians(theta))
    y = self.pose['y'] + (self.limts['size'] + 0) * math.sin(math.radians(theta))    
    pygame.draw.line(gameDisplay, (255,0,255), [int(self.pose['x']), int(self.pose['y'])], [int(x), int(y)], 10)

  def tick(self, display_width, display_height, game_image, reward):
    bullet = self.act(game_image, reward)
    super(Ship, self).tick(display_width, display_height)
    return bullet

  def act(self, game_image,reward):
    
    #do tensorflow stuff here
    action = self.deep_learner.get_keys_pressed(game_image,reward,reward!=0)

    if len(action) > 0:
      if action[0] == pygame.K_LEFT:
        # print('Left')
        self.rotateLeft()
      if action[0] == pygame.K_RIGHT:
        # print('Right')
        self.rotateRight()
      if action[0] == pygame.K_UP:
        # print('Up')
        self.accelerate()
      if action[0] == pygame.K_DOWN:
        # print('Down')
        self.deccelerate()
      if action[0] == pygame.K_SPACE:
        # print('Space')
        return self.fire()
    return None

  def checkCollision(self,sphere):
    if super(Ship, self).checkCollision(sphere):
      return True
    return False

#############################################################
class HumanShip(Ship):
  def __init__(self, x, y, t, max_vel, max_acc, vel=0):
    Ship.__init__(self, x, y, t, max_vel, max_acc, vel)

  def act(self, other_ship, bullets):
    #TODO add in human generated stratagy for the AI to learn against or from instead of a random action
    return None

#############################################################
class Bullet(Sphere):
  def __init__(self, x, y, t, max_vel, max_acc, vel=0):
    Sphere.__init__(self, x, y, t, max_vel, max_acc, 10, vel)

  def tick(self, display_width, display_height):
    super(Bullet, self).tick(display_width, display_height)

    if self.pose['x'] >= display_width or self.pose['x'] <= 0 or self.pose['y'] >= display_height or self.pose['y'] <= 0:
      return True
    return False

