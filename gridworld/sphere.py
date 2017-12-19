#!/usr/bin/env python
import math
import pygame
from math import sqrt
from learner import DeepLearner
from dqrnn import DQRNLearner

class Sphere(object):
  def __init__(self, x, y, size, color):
    self.limts = {}
    self.limts['size'] = size 
    self.color = color
    self.setValues(x, y)
    
  def setValues(self, x, y):
    self.pose = {}
    self.pose['x'] = x
    self.pose['y'] = y   

  def moveLeft(self, display_width, display_height):
    self.move(-self.limts['size'], 0, display_width, display_height)

  def moveRight(self, display_width, display_height):
    self.move(self.limts['size'], 0, display_width, display_height)
    
  def moveUp(self, display_width, display_height):
    self.move(0, self.limts['size'], display_width, display_height)
    

  def moveDown(self, display_width, display_height):
    self.move(0, -self.limts['size'], display_width, display_height)
    
  def move(self, dx, dy, width_limit, height_limit):
    self.pose['x'] += dx
    self.pose['y'] += dy

    if self.pose['x'] + self.limts['size'] > width_limit[1]:
      self.pose['x'] = width_limit[1] - self.limts['size']
    elif self.pose['x'] < width_limit[0]:
      self.pose['x'] = width_limit[0]

    if self.pose['y'] + self.limts['size'] > height_limit[1]:
      self.pose['y'] = height_limit[1] - self.limts['size']
    elif self.pose['y'] < height_limit[0]:
      self.pose['y'] =  height_limit[0]

  def render(self, gameDisplay):
    # pygame.draw.circle(gameDisplay, self.color, (int(self.pose['x']), int(self.pose['y'])), self.limts['size'])
    # pygame.draw.circle(gameDisplay, (0,0,255), (int(self.pose['x']), int(self.pose['y'])), self.limts['size']/2)
    x1 = self.pose['x']
    y1 = self.pose['y']
    pygame.draw.rect(gameDisplay,self.color,(x1,y1,self.limts['size'],self.limts['size']))

  def checkCollision(self,sphere):
    x_diff = self.pose['x'] - sphere.pose['x']
    y_diff = self.pose['y'] - sphere.pose['y']
    hyp = sqrt(x_diff**2 + y_diff**2)
    return hyp < (self.limts['size']/2 + sphere.limts['size']/2)

  def getPose(self):
    return self.pose, self.limts

#############################################################
class Ship(Sphere):
  def __init__(self, id, x, y, color):
    Sphere.__init__(self, x, y, 100, color)
    self.id = id
    # self.deep_learner = DeepLearner(str(id), checkpoint_path="deep_q_spaceshooter_networks_"+str(self.id))
    self.deep_learner = DQRNLearner(str(id), checkpoint_path="deep_q_spaceshooter_networks_"+str(self.id))

  def tick(self,game_image,reward,terminal,display_width,display_height):
    action = self.deep_learner.get_keys_pressed(game_image,reward,terminal)
    if action != None:
      if len(action) > 0:
        if action[0] == pygame.K_LEFT:
          self.moveLeft(display_width, display_height)
        elif action[0] == pygame.K_RIGHT:
          self.moveRight(display_width, display_height)
        elif action[0] == pygame.K_UP:
          self.moveUp(display_width, display_height)
        elif action[0] == pygame.K_DOWN:
          self.moveDown(display_width, display_height)

class AmmoBox(Sphere):
  def __init__(self, x, y):
    Sphere.__init__(self, x, y, 100, (0, 255, 0))
    
class Mine(Sphere):
  def __init__(self, x, y):
    Sphere.__init__(self, x, y, 100, (0, 0, 255))

