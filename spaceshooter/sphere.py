#!/usr/bin/env python
import math
import pygame

class Sphere(object):
  def __init__(self, x, y, t, max_vel, max_acc, size, vel=0):
    self.limts = {}
    self.limts['max_vel'] = max_vel
    self.limts['max_acc'] = max_acc
    self.limts['size'] = size

    self.pose = {}
    self.pose['x'] = x
    self.pose['y'] = y
    self.pose['t'] = t

    self.pose['vel_x'] = 0
    self.pose['vel_y'] = 0
    self.setVelocities(vel)
    
  def tick(self):
    self.move()

  def rotateLeft(self):
    self.rotate(-self.limts['max_acc'] * 10)

  def rotateRight(self):
    self.rotate(self.limts['max_acc'] * 10)

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

  def move(self):
    self.pose['x'] += self.pose['vel_x']
    self.pose['y'] += self.pose['vel_y']

  def setVelocities(self, acc):
    theta = self.pose['t']
    self.pose['vel_x'] += acc * math.cos(math.radians(theta))
    self.pose['vel_y'] += acc * math.sin(math.radians(theta))

    if self.pose['vel_x'] > self.limts['max_vel']:
        self.pose['vel_x'] = self.limts['max_vel']

    if self.pose['vel_y'] > self.limts['max_vel']:
        self.pose['vel_y'] = self.limts['max_vel']

  def render(self, gameDisplay):
    pygame.draw.circle(gameDisplay, (0,255,255), (int(self.pose['x']), int(self.pose['y'])), self.limts['size'])
    theta = self.pose['t']
    x = self.pose['x'] + self.limts['size'] * math.cos(math.radians(theta))
    y = self.pose['y'] + self.limts['size'] * math.sin(math.radians(theta))    
    pygame.draw.line(gameDisplay, (255,0,255), [int(self.pose['x']), int(self.pose['y'])], [int(x), int(y)], 5)

class Ship(Sphere):
  def __init__(self, x, y, t, max_vel, max_acc, vel=0):
    Sphere.__init__(self, x, y, t, max_vel, max_acc, 20, vel)

  def fire(self):
    x = self.pose['x']
    y = self.pose['y']
    t = self.pose['t']
    max_vel = 10
    max_acc = 0
    vel = 10
    bullet = Bullet(x, y, t, max_vel, max_acc, vel)
    return bullet

  def tick(self):
    super(Ship, self).tick()

class Bullet(Sphere):
  def __init__(self, x, y, t, max_vel, max_acc, vel=0):
    Sphere.__init__(self, x, y, t, max_vel, max_acc, 5, vel)
